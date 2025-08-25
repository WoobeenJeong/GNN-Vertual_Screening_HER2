#!/usr/bin/env python3

import os, re, glob, csv

SDF_DIR = "/home/ssm-user/project/gnina"
OUTFILE = "gnina_ranking.txt"

def parse_block(block):
    def find_prop(name):
        m = re.search(r'>\s*<%s>\s*\n\s*([^\n\r]+)' % re.escape(name), block, re.IGNORECASE)
        return m.group(1).strip() if m else None

    smiles = find_prop("SMILES") or find_prop("smiles") or find_prop("SMILE") or find_prop("SMI")
    min_aff = find_prop("minimizedAffinity")
    cnn_score = find_prop("CNNscore")
    cnn_aff = find_prop("CNNaffinity")

    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    return {
        "smiles": smiles,
        "minimizedAffinity": to_float(min_aff),
        "CNNscore": to_float(cnn_score),
        "CNNaffinity": to_float(cnn_aff),
    }

def best_pose_for_file(filepath):
    txt = open(filepath, "r", encoding="utf-8", errors="ignore").read()
    blocks = [b for b in re.split(r'\n\${4}\s*\n', txt) if b.strip()]
    parsed = [parse_block(b) for b in blocks]
    if not parsed:
        return None

    affs = [(i, p["minimizedAffinity"]) for i, p in enumerate(parsed)]
    affs_with = [ (i,a) for i,a in affs if a is not None ]
    if affs_with:
        best_idx = min(affs_with, key=lambda t: t[1])[0]
    else:
        scores = [(i, p["CNNscore"]) for i,p in enumerate(parsed)]
        scores_with = [ (i,s) for i,s in scores if s is not None ]
        if scores_with:
            best_idx = max(scores_with, key=lambda t: t[1])[0]
        else:
            best_idx = 0

    return parsed[best_idx]

def ligand_name_from_filename(fname):
    base = os.path.splitext(os.path.basename(fname))[0]
    base = re.sub(r'(_docked|-docked|_out|-out|\.docked)$','', base, flags=re.IGNORECASE)
    return base

def main():
    sdffiles = sorted(glob.glob(os.path.join(SDF_DIR, "*.sdf")))
    if not sdffiles:
        print("No .sdf files found in", SDF_DIR)
        return

    rows = []
    for f in sdffiles:
        best = best_pose_for_file(f)
        ligand = ligand_name_from_filename(f)
        if best is None:
            print("Warning: no blocks parsed for", f)
            continue
        rows.append({
            "Ligand": ligand,
            "SMILES": best.get("smiles") or "",
            "Binding Affinity (kcal/mol)": best.get("minimizedAffinity"),
            "CNNscore": best.get("CNNscore"),
            "CNNaffinity": best.get("CNNaffinity"),
            "source_file": f
        })

    def sort_key(r):
        a = r["Binding Affinity (kcal/mol)"]
        return (a if a is not None else 9999)
    rows_sorted = sorted(rows, key=sort_key)

    with open(OUTFILE, "w", newline='', encoding="utf-8") as fout:
        writer = csv.writer(fout, delimiter='\t')
        writer.writerow(["Rank","Ligand","SMILES","Binding Affinity (kcal/mol)","CNNscore","CNNaffinity"])
        for i, r in enumerate(rows_sorted, start=1):
            writer.writerow([
                i,
                r["Ligand"],
                r["SMILES"],
                ("{:.2f}".format(r["Binding Affinity (kcal/mol)"]) if r["Binding Affinity (kcal/mol)"] is not None else ""),
                ("" if r["CNNscore"] is None else "{:.6f}".format(r["CNNscore"])),
                ("" if r["CNNaffinity"] is None else "{:.6f}".format(r["CNNaffinity"]))
            ])

    print(f"Wrote ranking for {len(rows_sorted)} ligands -> {OUTFILE}")
    print("Top 10:")
    for i, r in enumerate(rows_sorted[:10], start=1):
        print(f"{i:2d}. {r['Ligand']}\t{r['Binding Affinity (kcal/mol)']}  CNNscore={r['CNNscore']}  CNNaffinity={r['CNNaffinity']}")

if __name__ == "__main__":
    main()
