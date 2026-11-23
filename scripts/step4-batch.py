import pandas as pd
import urllib.request
from pathlib import Path
from glycan_contacts import parse_contacts          # your existing script
from step3_fetch_variants import (                  # see rename note below
    build_variant_index, overlay_variants,
    load_glygen_germline, load_glygen_somatic
)

# ── Step A: build job list from Kate's CSV ────────────────────────────────────
kate = pd.read_csv("../data/downloads/human_proteoform_glycosylation_sites_rcsb_pdb.csv")
kate = kate[kate["saccharide"].notna() & (kate["saccharide"].str.strip() != "")]
kate["uniprot_id"] = kate["uniprotkb_canonical_ac"].str.split("-").str[0]
jobs = kate[["pdb_id", "uniprot_id", "saccharide"]].drop_duplicates()
pdb_uniprot_map = dict(zip(jobs["pdb_id"].str.upper(), jobs["uniprot_id"]))
uniprot_ids     = list(kate["uniprot_id"].unique())

downloads_dir = Path("../data/downloads")
downloads_dir.mkdir(parents=True, exist_ok=True)

# ── Step B: batch download + parse ───────────────────────────────────────────
all_frames, failed = [], []

for _, row in jobs.iterrows():
    pdb_id   = row["pdb_id"].upper()
    pdb_path = downloads_dir / f"{pdb_id}.pdb"

    if not pdb_path.exists():
        try:
            urllib.request.urlretrieve(
                f"https://files.rcsb.org/download/{pdb_id}.pdb", pdb_path)
        except Exception as e:
            failed.append((pdb_id, str(e))); continue

    try:
        df_c = parse_contacts(str(pdb_path), outer_cutoff=8.0)
        if not df_c.empty:
            df_c["uniprot_ac"] = row["uniprot_id"]
            all_frames.append(df_c)
    except Exception as e:
        failed.append((pdb_id, str(e)))

master = pd.concat(all_frames, ignore_index=True)
master.to_csv("../data/generated/step4/step4_all_contacts.csv", index=False)
pd.DataFrame(failed, columns=["pdb_id","error"]).to_csv("step4_failed.csv", index=False)

# ── Step C: variant overlay (calls step3 functions directly) ──────────────────
glygen_germline = load_glygen_germline("../data/downloads/human_protein_mutation_germline_all.csv")
glygen_somatic  = load_glygen_somatic("../data/downloads/human_protein_mutation_cancer_all.csv")

gene_map = {}   # optional: {"P01857": "IGHG1", "P06729": "CD2", ...}

# Apply SIFTS offsets so PDB residue numbers align with UniProt positions
from step3_fetch_variants import fetch_sifts_offset

for pdb_id in master["pdb_id"].unique():
    for chain in master[master["pdb_id"] == pdb_id]["protein_chain"].unique():
        offset = fetch_sifts_offset(pdb_id, chain)
        mask = (master["pdb_id"] == pdb_id) & (master["protein_chain"] == chain)
        master.loc[mask, "protein_resnum"] -= offset

vindex   = build_variant_index(uniprot_ids, gene_map, glygen_germline, glygen_somatic)
result   = overlay_variants(master, vindex, pdb_uniprot_map)
result.to_csv("../data/generated/step4/step4_dataset.csv", index=False)