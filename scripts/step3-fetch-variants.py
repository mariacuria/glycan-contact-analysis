# python step3-fetch-variants.py --use_sifts --contacts /data/shared/repos/non-covalent-glycan-interactions/data/generated/step2/step2_all_contacts.csv --uniprot P01857,P06729 --pdb_uniprot_map 3AVE:P01857,2WAH:P01857,1GYA:P06729 --out /data/shared/repos/non-covalent-glycan-interactions/data/generated/step3/step3_dataset.csv --glygen_somatic /data/shared/repos/non-covalent-glycan-interactions/data/downloads/human_protein_mutation_cancer_all.csv --glygen_germline /data/shared/repos/non-covalent-glycan-interactions/data/downloads/human_protein_mutation_germline_all.csv
"""
step3-fetch-variants.py
=================
Step 3 of thesis pipeline: pull germline (ClinVar + UniProt) and somatic
(COSMIC) missense variants for a given protein, then overlay them onto the
Step 2 contact table by matching protein residue numbers.

Sources
-------
  Germline:
    1. UniProt REST API  — natural variants (disease-associated missense)
       No key required. Returns position, wt/mut aa, disease association.
    2. NCBI ClinVar E-utilities — pathogenic / likely-pathogenic missense
       No key required (rate-limited; add NCBI_API_KEY env var for 10 req/s).

  Somatic:
    3. BioMuta (biomuta_v6.1.csv)

Output columns added to the Step 2 contact CSV
-----------------------------------------------
  has_germline_variant   : bool
  germline_sources       : pipe-separated list (UniProt | ClinVar | both)
  germline_variants      : JSON list of {pos, wt, mut, disease, source}
  has_somatic_variant    : bool
  somatic_variants       : JSON list of {pos, wt, mut, cancer_type, cosmic_id}
  is_glycan_site_residue : bool  (True if protein_resnum == glycan Asn/Ser/Thr)
  sequon_gain_loss       : str   (gain | loss | none) — only for N-X-S/T sites

Usage
-----
    pip install biopython pandas requests tqdm

    # Minimal (germline only):
    python fetch_variants.py \
        --contacts step2_all_contacts.csv \
        --uniprot  P01857,P06729 \
        --out      step3_dataset.csv

    # With COSMIC (register at cancer.sanger.ac.uk → get token):
    COSMIC_TOKEN=<base64_token> python fetch_variants.py \
        --contacts step2_all_contacts.csv \
        --uniprot  P01857,P06729 \
        --out      step3_dataset.csv

    # Reproduce thesis demo (3AVE = P01857, 2WAH = P01857, 1GYA = P06729):
    python fetch_variants.py \
        --contacts step2_all_contacts.csv \
        --uniprot  P01857,P06729 \
        --pdb_uniprot_map 3AVE:P01857,1GYA:P06729 \
        --out step3_dataset.csv
"""

import argparse
import json
import os
import re
import time
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────
UNIPROT_API  = "https://rest.uniprot.org/uniprotkb"
CLINVAR_API  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
COSMIC_API   = "https://cancer.sanger.ac.uk/api/cosmic/v3.1"

NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")   # optional, raises rate limit
COSMIC_TOKEN = os.environ.get("COSMIC_TOKEN", "")   # required for somatic data

# Amino acid 1→3 letter conversion
AA1TO3 = {
    "A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE","G":"GLY","H":"HIS",
    "I":"ILE","K":"LYS","L":"LEU","M":"MET","N":"ASN","P":"PRO","Q":"GLN",
    "R":"ARG","S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR","*":"TER",
}
AA3TO1 = {v: k for k, v in AA1TO3.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 1. UniProt — natural variants (germline)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_uniprot_variants(uniprot_id: str) -> list[dict]:
    """
    Pull missense natural variants from UniProt features endpoint.
    Returns list of dicts: {pos, wt, mut, disease, source='UniProt'}.
    """
    url = f"{UNIPROT_API}/{uniprot_id}.json"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  [UniProt] {uniprot_id} fetch failed: {e}")
        return []

    data = r.json()
    variants = []
    for feat in data.get("features", []):
        if feat.get("type") not in ("Natural variant", "Mutagenesis"):
            continue
        loc = feat.get("location", {})
        start = loc.get("start", {}).get("value")
        end   = loc.get("end",   {}).get("value")
        if start is None or start != end:
            continue  # skip range features, keep single-residue only

        desc = feat.get("description", "")
        # Parse "X -> Y" or "X -> Y; disease name"
        alt_match = re.search(r"([A-Z*])\s*->\s*([A-Z*])", desc)
        if not alt_match:
            continue
        wt, mut = alt_match.group(1), alt_match.group(2)

        disease = ""
        if ";" in desc:
            disease = desc.split(";", 1)[1].strip()
        elif "in " in desc.lower():
            disease = re.sub(r".*?\bin\b\s*", "", desc, flags=re.IGNORECASE).strip()

        variants.append({
            "pos"     : int(start),
            "wt"      : wt,
            "mut"     : mut,
            "disease" : disease,
            "source"  : "UniProt",
        })

    print(f"  [UniProt] {uniprot_id}: {len(variants)} variants retrieved")
    return variants


# ─────────────────────────────────────────────────────────────────────────────
# 2. ClinVar — pathogenic / likely-pathogenic (germline)
#    GlyGen germline
# ─────────────────────────────────────────────────────────────────────────────

def fetch_clinvar_variants(gene_symbol: str,
                           max_results: int = 500) -> list[dict]:
    """
    Query ClinVar via NCBI E-utilities for pathogenic missense variants.
    gene_symbol: HGNC symbol, e.g. 'IGHG1', 'CD2', 'NAGLU', 'SGSH'.
    """
    params = {
        "db"      : "clinvar",
        "term"    : (f"{gene_symbol}[gene] AND "
                     "(pathogenic[clinsig] OR likely pathogenic[clinsig]) AND "
                     "missense[moltype]"),
        "retmax"  : max_results,
        "retmode" : "json",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    # esearch
    try:
        r = requests.get(f"{CLINVAR_API}/esearch.fcgi", params=params, timeout=20)
        r.raise_for_status()
        ids = r.json()["esearchresult"].get("idlist", [])
    except Exception as e:
        print(f"  [ClinVar] esearch failed for {gene_symbol}: {e}")
        return []

    if not ids:
        print(f"  [ClinVar] {gene_symbol}: no pathogenic variants found")
        return []

    # esummary (batch of up to 500)
    time.sleep(0.35 if not NCBI_API_KEY else 0.11)
    params2 = {
        "db"      : "clinvar",
        "id"      : ",".join(ids),
        "retmode" : "json",
    }
    if NCBI_API_KEY:
        params2["api_key"] = NCBI_API_KEY
    try:
        r2 = requests.get(f"{CLINVAR_API}/esummary.fcgi", params=params2, timeout=30)
        r2.raise_for_status()
        result = r2.json().get("result", {})
    except Exception as e:
        print(f"  [ClinVar] esummary failed: {e}")
        return []

    variants = []
    for uid in ids:
        entry = result.get(uid, {})
        # Extract protein change e.g. "p.Arg301His"
        protein_change = ""
        for alias in entry.get("variation_set", [{}])[0].get("cdna_change", ""):
            pass
        # Title field usually contains "p.XnnY"
        title = entry.get("title", "")
        p_match = re.search(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter|\*)", title)
        if not p_match:
            continue
        wt3, pos_str, mut3 = p_match.group(1), p_match.group(2), p_match.group(3)
        wt  = AA3TO1.get(wt3.upper(),  wt3)
        mut = AA3TO1.get(mut3.upper(), mut3)
        conditions = "; ".join(
            t.get("trait_name", "") for t in
            entry.get("trait_set", [{}])[0].get("trait_xrefs", [{}])
        ) or entry.get("title", "")

        variants.append({
            "pos"     : int(pos_str),
            "wt"      : wt,
            "mut"     : mut,
            "disease" : conditions,
            "source"  : "ClinVar",
        })

    print(f"  [ClinVar] {gene_symbol}: {len(variants)} variants parsed")
    return variants

def load_glygen_germline(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=",")
    df["uniprotkb_canonical_ac"] = df["uniprotkb_canonical_ac"].str.split("-").str[0]
    return df

def fetch_glygen_germline_variants(uniprot_id: str,
                                   glygen_germline: pd.DataFrame) -> list[dict]:
    subset = glygen_germline[glygen_germline["uniprotkb_canonical_ac"] == uniprot_id]
    variants = []
    for _, row in subset.iterrows():
        variants.append({
            "pos": int(row["aa_pos"]),
            "wt": row["ref_aa"],
            "mut": row["alt_aa"],
            "disease": str(row.get("disease_description", "")),
            "source": "GlyGen-Germline",
        })
    seen = set()
    deduped = []
    for v in variants:
        key = (v["pos"], v["wt"], v["mut"])
        if key not in seen:
            seen.add(key)
            deduped.append(v)
    variants = deduped
    print(f"  [GlyGen-Germline] {uniprot_id}: {len(variants)} variants")
    return variants

# ─────────────────────────────────────────────────────────────────────────────
# 3. BioMuta — somatic mutations
# ─────────────────────────────────────────────────────────────────────────────

def load_glygen_somatic(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["uniprotkb_canonical_ac"] = df["uniprotkb_canonical_ac"].str.split("-").str[0]
    return df

def fetch_cosmic_variants(gene_symbol: str, max_results: int = 500,
                          glygen_somatic: pd.DataFrame = None) -> list[dict]:
    if glygen_somatic is not None:
        subset = glygen_somatic[glygen_somatic["uniprotkb_canonical_ac"] == gene_symbol]
        variants = []
        for _, row in subset.iterrows():
            variants.append({
                "pos": int(row["aa_pos"]),
                "wt": row["ref_aa"],
                "mut": row["alt_aa"],
                "cancer_type": row["do_name"],
                "cosmic_id": row.get("dbsnp_id", ""),
                "source": "GlyGen-BioMuta",
            })
        print(f"  [GlyGen-Somatic] {gene_symbol}: {len(variants)} variants")
        return variants


# ─────────────────────────────────────────────────────────────────────────────
# 4. N-linked sequon analysis
# ─────────────────────────────────────────────────────────────────────────────

def check_sequon(wt_aa: str, mut_aa: str,
                 pos: int, sequence: Optional[str]) -> str:
    """
    Determine if a missense variant creates (gain) or destroys (loss)
    an N-X-S/T sequon, where X ≠ Pro.
    Returns 'gain', 'loss', or 'none'.
    """
    if sequence is None:
        return "unknown"
    seq_len = len(sequence)
    idx = pos - 1  # 0-based

    def is_sequon_start(i):
        """Is there a valid N-X-S/T starting at position i?"""
        if i < 0 or i + 2 >= seq_len:
            return False
        return (sequence[i] == "N" and
                sequence[i + 1] != "P" and
                sequence[i + 2] in ("S", "T"))

    # Check if the mutation affects sequon at pos, pos-1, or pos-2
    for offset in range(3):
        n_pos = idx - offset
        if n_pos < 0:
            continue
        # Build mutant sequence locally
        mut_seq = list(sequence)
        mut_seq[idx] = mut_aa
        mut_seq = "".join(mut_seq)

        wt_has  = is_sequon_start(n_pos) if sequence[n_pos] == "N" else (
                      n_pos + 1 == idx and sequence[n_pos] == "N" or
                      n_pos + 2 == idx and sequence[n_pos] == "N")
        # Simplified: just check the exact triplet starting at n_pos
        wt_has  = (sequence[n_pos:n_pos+3] if n_pos+3 <= seq_len else "") 
        mut_has = (mut_seq[n_pos:n_pos+3]  if n_pos+3 <= seq_len else "")

        def valid_sequon(s):
            return (len(s) == 3 and s[0] == "N" and s[1] != "P"
                    and s[2] in ("S", "T"))

        if valid_sequon(wt_has) and not valid_sequon(mut_has):
            return "loss"
        if not valid_sequon(wt_has) and valid_sequon(mut_has):
            return "gain"

    return "none"


def fetch_uniprot_sequence(uniprot_id: str) -> Optional[str]:
    """Fetch canonical FASTA sequence from UniProt."""
    url = f"{UNIPROT_API}/{uniprot_id}.fasta"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        return "".join(l for l in lines if not l.startswith(">"))
    except Exception as e:
        print(f"  [UniProt] Sequence fetch failed for {uniprot_id}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main overlay logic
# ─────────────────────────────────────────────────────────────────────────────

def build_variant_index(uniprot_ids: list[str],
                        gene_map: dict[str, str],
                        glygen_germline: pd.DataFrame = None,
                        glygen_somatic: pd.DataFrame = None) -> dict[str, dict]:
    """
    For each UniProt ID, fetch all variants and the canonical sequence.
    Returns {uniprot_id: {germline: [...], somatic: [...], sequence: str}}.
    """
    index = {}
    for uid in uniprot_ids:
        gene = gene_map.get(uid, uid)
        print(f"\nFetching variants for {uid} (gene: {gene})")
        germline = fetch_uniprot_variants(uid)
        germline += fetch_glygen_germline_variants(uid, glygen_germline)
        germline += fetch_clinvar_variants(gene)
        somatic  = fetch_cosmic_variants(gene, glygen_somatic=glygen_somatic)
        seq      = fetch_uniprot_sequence(uid)
        index[uid] = {"germline": germline, "somatic": somatic, "sequence": seq}
    return index


def overlay_variants(contacts_df: pd.DataFrame,
                     variant_index: dict,
                     pdb_uniprot_map: dict[str, str]) -> pd.DataFrame:
    """
    Join variant data onto the Step 2 contact table.
    pdb_uniprot_map: {'3AVE': 'P01857', '1GYA': 'P06729', ...}
    """
    # Initialise new columns
    contacts_df = contacts_df.copy()
    new_cols = [
        "uniprot_id", "has_germline_variant", "germline_sources",
        "germline_variants", "has_somatic_variant", "somatic_variants",
        "is_glycan_site_residue", "sequon_gain_loss",
    ]
    for col in new_cols:
        contacts_df[col] = None

    for idx, row in tqdm(contacts_df.iterrows(),
                         total=len(contacts_df), desc="Overlaying variants"):
        uid = pdb_uniprot_map.get(row["pdb_id"], None)
        if uid is None or uid not in variant_index:
            continue

        vdata   = variant_index[uid]
        res_pos = int(row["protein_resnum"])
        seq     = vdata["sequence"]

        # Map PDB residue number to UniProt position
        # NOTE: If the PDB numbering differs from UniProt (e.g. 3AVE Asn297 = UniProt Asn180),
        # you need to apply the offset here.  The mapping can be read from SIFTS XML
        # (see helper fetch_sifts_offset below).  For the demo we use the raw PDB number;
        # apply offset manually or via fetch_sifts_offset().
        uniprot_pos = res_pos  # replace with (res_pos - offset) if needed

        # Germline
        g_hits = [v for v in vdata["germline"] if v["pos"] == uniprot_pos]
        sources = list({v["source"] for v in g_hits})

        # Sequon gain/loss for each germline variant
        for v in g_hits:
            v["sequon"] = check_sequon(v["wt"], v["mut"], v["pos"], seq)

        # Somatic
        s_hits = [v for v in vdata["somatic"] if v["pos"] == uniprot_pos]

        # Is this residue the glycan-bearing Asn/Ser/Thr?
        prot_resname = row["protein_resname"].upper()
        is_glycan_site = prot_resname in ("ASN", "SER", "THR") and (
            row.get("interaction_type") in (None, "") or True
        )
        # More precise: glycan_chain == protein_chain and position is
        # directly under the NAG/GNX attachment — use sequon check
        if seq and uniprot_pos > 0 and uniprot_pos <= len(seq):
            aa = seq[uniprot_pos - 1]
            is_glycan_site = (aa == "N" and
                              check_sequon("N", "N", uniprot_pos, seq) == "none"
                              and uniprot_pos + 2 <= len(seq)
                              and seq[uniprot_pos + 1] != "P"
                              and seq[uniprot_pos + 2] in ("S", "T"))

        # Sequon gain/loss summary for this position (best call across variants)
        sequon_calls = [v.get("sequon", "none") for v in g_hits]
        sequon_summary = (
            "gain" if "gain" in sequon_calls else
            "loss" if "loss" in sequon_calls else
            "none"
        )

        contacts_df.at[idx, "uniprot_id"]          = uid
        contacts_df.at[idx, "has_germline_variant"] = len(g_hits) > 0
        contacts_df.at[idx, "germline_sources"]     = " | ".join(sources) if sources else ""
        contacts_df.at[idx, "germline_variants"]    = json.dumps(g_hits)
        contacts_df.at[idx, "has_somatic_variant"]  = len(s_hits) > 0
        contacts_df.at[idx, "somatic_variants"]     = json.dumps(s_hits)
        contacts_df.at[idx, "is_glycan_site_residue"] = is_glycan_site
        contacts_df.at[idx, "sequon_gain_loss"]     = sequon_summary

    return contacts_df


# ─────────────────────────────────────────────────────────────────────────────
# Optional: SIFTS offset helper (PDB residue → UniProt residue mapping)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_sifts_offset(pdb_id: str, chain: str) -> int:
    """
    Query the SIFTS API to get the offset between PDB and UniProt numbering.
    Returns integer offset such that: uniprot_pos = pdb_resnum - offset.
    Falls back to 0 (no offset) on failure.

    Example: 3AVE chain A — Asn297 (PDB) = Asn180 (UniProt), offset = 117.
    """
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json().get(pdb_id.lower(), {}).get("UniProt", {})
        for uid_key, mapping in data.items():
            for seg in mapping.get("mappings", []):
                if seg.get("chain_id") == chain:
                    pdb_start    = seg["start"]["residue_number"]
                    uniprot_start = seg["unp_start"]
                    return pdb_start - uniprot_start
    except Exception as e:
        print(f"  [SIFTS] Offset fetch failed for {pdb_id}/{chain}: {e}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Overlay germline + somatic variants on Step 2 contact table.")
    ap.add_argument("--contacts", required=True,
                    help="Step 2 output CSV (step2_all_contacts.csv)")
    ap.add_argument("--glygen_somatic",  default="", help="Path to biomuta_v6.1.csv")
    ap.add_argument("--glygen_germline", default="/data/shared/repos/non-covalent-glycan-interactions/data/downloads/human_protein_mutation_germline_all.csv", help="Path to human_protein_mutation_germline_all.csv")
    ap.add_argument("--uniprot", required=True,
                    help="Comma-separated UniProt IDs, e.g. P01857,P06729")
    ap.add_argument("--gene_map", default="",
                    help="UniProt→gene map e.g. P01857:IGHG1,P06729:CD2")
    ap.add_argument("--pdb_uniprot_map", default="",
                    help="PDB→UniProt map e.g. 3AVE:P01857,1GYA:P06729")
    ap.add_argument("--use_sifts", action="store_true",
                    help="Auto-fetch PDB↔UniProt residue offsets via SIFTS")
    ap.add_argument("--out", default="step3_dataset.csv")
    args = ap.parse_args()

    uniprot_ids = [u.strip() for u in args.uniprot.split(",")]

    gene_map = {}
    for pair in args.gene_map.split(","):
        if ":" in pair:
            uid, gene = pair.split(":", 1)
            gene_map[uid.strip()] = gene.strip()

    pdb_uniprot_map = {}
    for pair in args.pdb_uniprot_map.split(","):
        if ":" in pair:
            pdb, uid = pair.split(":", 1)
            pdb_uniprot_map[pdb.strip().upper()] = uid.strip()

    contacts = pd.read_csv(args.contacts)
    print(f"Loaded {len(contacts)} contact rows from {args.contacts}")

    # Optional SIFTS offset per PDB
    if args.use_sifts:
        for pdb_id, uid in pdb_uniprot_map.items():
            chains = contacts[contacts["pdb_id"] == pdb_id]["protein_chain"].unique()
            for chain in chains:
                offset = fetch_sifts_offset(pdb_id, chain)
                print(f"  SIFTS offset {pdb_id}/{chain}: {offset} "
                      f"(pdb_resnum - {offset} = uniprot_pos)")
                # Apply offset — create a temporary mapped column
                mask = ((contacts["pdb_id"] == pdb_id) &
                        (contacts["protein_chain"] == chain))
                contacts.loc[mask, "protein_resnum_uniprot"] = (
                    contacts.loc[mask, "protein_resnum"] - offset
                )
        # Use remapped column if SIFTS was applied
        if "protein_resnum_uniprot" in contacts.columns:
            contacts["protein_resnum"] = contacts["protein_resnum_uniprot"].fillna(
                contacts["protein_resnum"])

    glygen_somatic  = load_glygen_somatic(args.glygen_somatic)   if args.glygen_somatic  else None
    glygen_germline = load_glygen_germline(args.glygen_germline) if args.glygen_germline else None

    vindex = build_variant_index(uniprot_ids, gene_map, glygen_germline, glygen_somatic)
    result = overlay_variants(contacts, vindex, pdb_uniprot_map)

    result.to_csv(args.out, index=False)
    print(f"\nStep 3 complete → {args.out}  ({len(result)} rows)")

    # Quick summary
    gl = result["has_germline_variant"].sum()
    so = result["has_somatic_variant"].sum()
    gs = result["sequon_gain_loss"].value_counts().to_dict()
    print(f"\n  Residues with germline variant: {gl}")
    print(f"  Residues with somatic variant:  {so}")
    print(f"  Sequon events: {gs}")


if __name__ == "__main__":
    main()
