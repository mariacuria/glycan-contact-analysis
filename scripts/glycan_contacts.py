# Usage:
#   pyenv global 3.14.4
#   python glycan_contacts.py --pdb ../data/downloads/3AVE.pdb --cutoff 8.0 --out ../data/generated/3AVE_contacts.csv
#   python glycan_contacts.py --pdb ../data/downloads/1GYA.pdb --cutoff 8.0 --out ../data/generated/1GYA_contacts.csv
#   python glycan_contacts.py --pdb ../data/downloads/2WAH.pdb --cutoff 8.0 --out ../data/generated/2WAH_contacts.csv

import argparse
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, NeighborSearch, is_aa
from Bio.PDB.DSSP import DSSP  # optional – only for secondary structure annotation

warnings.simplefilter("ignore", BiopythonWarning)

# ── Glycan residue CCD codes to recognise in HETATM records ──────────────────
GLYCAN_CODES = {
    # N-linked core
    "NAG", "NDG",       # GlcNAc (β and α anomers)
    "BMA",              # β-D-mannose (core branching)
    "MAN",              # α-D-mannose
    "FUC", "FUL",       # α- and β-L-fucose
    # Antennae
    "GAL", "GLA",       # β- and α-D-galactose
    "SIA",              # N-acetylneuraminic acid (sialic acid)
    # Glucose forms (ER processing intermediates / glycolipids)
    "GLC", "BGC",
    # GAG / proteoglycan
    "XYL",              # xylose (GAG initiator on Ser)
    "GCU",              # D-glucuronic acid
    "IDR",              # L-iduronic acid (heparan sulfate)
    "GNX",              # GalNAc (O-linked initiator, mucin)
}

# ── Atom-level properties for interaction classification ──────────────────────
HBOND_DONORS    = {"N", "O", "S"}   # heavy-atom donor elements
HBOND_ACCEPTORS = {"N", "O", "S"}

CHARGED_POS = {                     # positively charged residue atoms
    "ARG": {"NE", "NH1", "NH2"},
    "LYS": {"NZ"},
    "HIS": {"ND1", "NE2"},          # treat His as potentially +
}
CHARGED_NEG = {                     # negatively charged residue atoms
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "SIA": {"O1A", "O1B"},          # carboxylate oxygens of sialic acid
}

AROMATIC_RES = {"PHE", "TYR", "TRP", "HIS"}
AROMATIC_RING_ATOMS = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
}

# Distance cutoffs (Å)
CUT_HBOND        = 3.5   # heavy-atom donor–acceptor (no explicit H) Pauling, L. (1960). The Nature of the Chemical Bond (3rd ed.). Cornell University.
CUT_ELECTRO      = 6.0   # charged pair
CUT_CHPI         = 4.5   # glycan C → aromatic centroid
CUT_VDW          = 4.0   # generic van der Waals contact Bondi, A. (1964). van der Waals Volumes and Radii. J. Phys. Chem.
CUT_HYDROPHOBIC  = 5.0   # C–C non-polar


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def is_glycan(residue) -> bool:
    return residue.get_resname().strip() in GLYCAN_CODES


def is_protein_residue(residue) -> bool:
    return is_aa(residue, standard=True)


def ring_centroid(residue) -> np.ndarray | None:
    """Return centroid of the aromatic ring atoms if available."""
    resname = residue.get_resname().strip()
    if resname not in AROMATIC_RING_ATOMS:
        return None
    coords = []
    for aname in AROMATIC_RING_ATOMS[resname]:
        try:
            coords.append(residue[aname].get_vector().get_array())
        except KeyError:
            pass
    if len(coords) < 3:
        return None
    return np.mean(coords, axis=0)


def classify_interaction(
    glycan_atom,
    prot_atom,
    prot_residue,
    dist: float,
    prot_ring_centroid=None,
) -> str:
    """
    Assign the most specific non-covalent interaction type for a given
    glycan_atom–protein_atom pair.
    Returns one of: 'H-bond', 'Electrostatic', 'CH-pi', 'Hydrophobic',
    'van der Waals', or None (skip water / covalent-bonded pairs).
    """
    ga_elem  = glycan_atom.element.strip().upper() if glycan_atom.element else glycan_atom.get_name()[0]
    pa_elem  = prot_atom.element.strip().upper()   if prot_atom.element   else prot_atom.get_name()[0]
    ga_name  = glycan_atom.get_name().strip()
    pa_name  = prot_atom.get_name().strip()
    pr_name  = prot_residue.get_resname().strip()

    # 1. Hydrogen bond: both atoms are N/O/S
    if ga_elem in HBOND_ACCEPTORS and pa_elem in HBOND_DONORS and dist <= CUT_HBOND:
        return "H-bond"
    if pa_elem in HBOND_ACCEPTORS and ga_elem in HBOND_DONORS and dist <= CUT_HBOND:
        return "H-bond"

    # 2. Electrostatic: charged pair
    is_glycan_neg = (glycan_atom.get_parent().get_resname().strip() == "SIA"
                     and ga_name in {"O1A", "O1B"})
    is_prot_pos   = pr_name in CHARGED_POS and pa_name in CHARGED_POS.get(pr_name, set())
    is_prot_neg   = pr_name in CHARGED_NEG and pa_name in CHARGED_NEG.get(pr_name, set())
    is_glycan_oh  = ga_elem == "O"  # hydroxyl / ring oxygen as weak acceptor

    if dist <= CUT_ELECTRO:
        if (is_glycan_neg and is_prot_pos) or (is_glycan_oh and is_prot_pos):
            return "Electrostatic"
        if is_prot_neg and ga_elem == "O":
            return "Electrostatic"

    # 3. CH–pi: glycan carbon close to aromatic ring centroid
    if ga_elem == "C" and prot_ring_centroid is not None:
        ga_coord = glycan_atom.get_vector().get_array()
        ch_pi_dist = float(np.linalg.norm(ga_coord - prot_ring_centroid))
        if ch_pi_dist <= CUT_CHPI:
            return "CH-pi"

    # 4. Hydrophobic: C–C contact, both non-polar
    if ga_elem == "C" and pa_elem == "C" and dist <= CUT_HYDROPHOBIC:
        # exclude carbonyl carbons (they are polar)
        if ga_name not in {"C", "C1", "C2"} and pa_name not in {"C", "CA"}:
            return "Hydrophobic"

    # 5. Generic van der Waals (catch-all for heavy atoms in range)
    if dist <= CUT_VDW and ga_elem != "H" and pa_elem != "H":
        return "van der Waals"

    return None  # outside all cutoffs


# ─────────────────────────────────────────────────────────────────────────────
# Core parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_contacts(pdb_path: str, outer_cutoff: float = 8.0) -> pd.DataFrame:
    """
    Parse one PDB file and return a DataFrame of all glycan–protein
    non-covalent contacts within outer_cutoff Angstroms.
    """
    pdb_id = Path(pdb_path).stem.upper()
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)
    model = structure[0]

    # Collect all atoms for neighbour search
    all_atoms = list(model.get_atoms())
    ns = NeighborSearch(all_atoms)

    # Identify glycan and protein residues
    glycan_residues  = [r for r in model.get_residues() if is_glycan(r)]
    protein_residues = {r.get_full_id(): r for r in model.get_residues() if is_protein_residue(r)}

    if not glycan_residues:
        print(f"[{pdb_id}] WARNING: No glycan residues found with known CCD codes.")
        return pd.DataFrame()

    print(f"[{pdb_id}] Found {len(glycan_residues)} glycan residues, "
          f"{len(protein_residues)} protein residues.")

    # Pre-compute ring centroids for aromatic residues
    ring_centroids = {}
    for fid, res in protein_residues.items():
        c = ring_centroid(res)
        if c is not None:
            ring_centroids[fid] = c

    rows = []
    seen_pairs = set()  # avoid duplicate residue-level rows from multiple atoms

    for g_res in glycan_residues:
        g_chain   = g_res.get_parent().get_id()
        g_resname = g_res.get_resname().strip()
        g_resnum  = g_res.get_id()[1]

        for g_atom in g_res.get_atoms():
            if g_atom.element and g_atom.element.strip() == "H":
                continue  # skip hydrogens

            # Find all protein atoms within outer cutoff
            neighbours = ns.search(g_atom.get_vector().get_array(), outer_cutoff, "A")

            for p_atom in neighbours:
                p_res = p_atom.get_parent()
                if not is_protein_residue(p_res):
                    continue

                p_fid = p_res.get_full_id()
                dist  = float(g_atom - p_atom)

                # Skip if this is a covalent glycan–Asn bond (N-glycosylation)
                # The Asn ND2 – NAG C1 bond is ~1.45 Å; exclude < 1.8 Å
                if dist < 1.8:
                    continue

                centroid = ring_centroids.get(p_fid)
                itype = classify_interaction(g_atom, p_atom, p_res, dist, centroid)
                if itype is None:
                    continue

                # Build a unique per-residue key for deduplication at residue level
                pair_key = (g_resname, g_resnum, g_chain,
                            p_res.get_resname().strip(), p_res.get_id()[1],
                            p_res.get_parent().get_id(), itype)

                # Record atom-level detail (keep closest atom pair per residue pair)
                rows.append({
                    "pdb_id"           : pdb_id,
                    "glycan_chain"     : g_chain,
                    "glycan_resname"   : g_resname,
                    "glycan_resnum"    : g_resnum,
                    "glycan_atom"      : g_atom.get_name().strip(),
                    "glycan_elem"      : g_atom.element.strip() if g_atom.element else "",
                    "protein_chain"    : p_res.get_parent().get_id(),
                    "protein_resname"  : p_res.get_resname().strip(),
                    "protein_resnum"   : p_res.get_id()[1],
                    "protein_atom"     : p_atom.get_name().strip(),
                    "protein_elem"     : p_atom.element.strip() if p_atom.element else "",
                    "distance_A"       : round(dist, 3),
                    "interaction_type" : itype,
                })

    if not rows:
        print(f"[{pdb_id}] No contacts found — check CCD codes or cutoff.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Keep closest-atom row per glycan_residue–protein_residue–interaction_type triple
    df = (df.sort_values("distance_A")
            .drop_duplicates(subset=["glycan_resname", "glycan_resnum", "glycan_chain",
                                     "protein_resname", "protein_resnum", "protein_chain",
                                     "interaction_type"])
            .reset_index(drop=True))

    print(f"[{pdb_id}] {len(df)} unique glycan–residue contact pairs identified.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Summary / annotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def summarise(df: pd.DataFrame) -> None:
    """Print a readable summary of the contact table."""
    if df.empty:
        return
    pdb = df["pdb_id"].iloc[0]
    print(f"\n{'='*60}")
    print(f" Contact summary for {pdb}")
    print(f"{'='*60}")
    print(f"\n Interaction type counts:")
    print(df["interaction_type"].value_counts().to_string())

    print(f"\n Glycan residue coverage:")
    print(df.groupby(["glycan_resname", "glycan_resnum", "glycan_chain"])
            ["interaction_type"].count()
            .rename("n_contacts")
            .to_string())

    print(f"\n Top 10 closest contacts:")
    top = df.nsmallest(10, "distance_A")[
        ["glycan_resname", "glycan_resnum", "protein_resname",
         "protein_resnum", "interaction_type", "distance_A"]
    ]
    print(top.to_string(index=False))


def add_secondary_structure(df: pd.DataFrame, pdb_path: str,
                            dssp_exe: str = "mkdssp") -> pd.DataFrame:
    """
    Optionally annotate each protein residue with its secondary structure
    (H = helix, E = strand, C = coil).  Requires DSSP installed.
    Falls back gracefully if DSSP is unavailable.
    """
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.DSSP import DSSP as _DSSP
        parser = PDBParser(QUIET=True)
        struct  = parser.get_structure("X", pdb_path)
        model   = struct[0]
        dssp    = _DSSP(model, pdb_path, dssp=dssp_exe)

        def get_ss(row):
            key = (row["protein_chain"], (" ", row["protein_resnum"], " "))
            try:
                return dssp[key][2]
            except KeyError:
                return "?"

        df["secondary_structure"] = df.apply(get_ss, axis=1)
    except Exception as e:
        print(f"  DSSP annotation skipped: {e}")
        df["secondary_structure"] = "?"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Parse PDB for glycan–protein non-covalent contacts.")
    ap.add_argument("--pdb",     required=True,  help="Path to PDB file")
    ap.add_argument("--cutoff",  type=float, default=8.0,
                    help="Outer distance shell in Å (default 8.0)")
    ap.add_argument("--out",     default=None,
                    help="Output CSV path (default: <pdb_id>_contacts.csv)")
    ap.add_argument("--dssp",    default=None,
                    help="Path to mkdssp executable (optional, for SS annotation)")
    ap.add_argument("--summary", action="store_true",
                    help="Print contact summary to stdout")
    args = ap.parse_args()

    out_path = args.out or Path(args.pdb).stem + "_contacts.csv"

    df = parse_contacts(args.pdb, outer_cutoff=args.cutoff)

    if not df.empty:
        if args.dssp:
            df = add_secondary_structure(df, args.pdb, dssp_exe=args.dssp)
        df.to_csv(out_path, index=False)
        print(f"\nSaved → {out_path}  ({len(df)} rows)")
        if args.summary:
            summarise(df)


if __name__ == "__main__":
    main()
