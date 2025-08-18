import csv
import sys
import os
import re

csv_path = sys.argv[1]
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

def safe_name(s):
    s = re.sub(r'[^A-Za-z0-9_\-\.]', '_', s)
    return s[:120]

with open(csv_path, newline='') as fh:
    sample = fh.read(8192)
    fh.seek(0)
    dialect = csv.Sniffer().sniff(sample)
    reader = csv.reader(fh, dialect)
    header = next(reader)

    cols = [h.lower() for h in header]
    smi_col = None
    id_col = None
    for idx, h in enumerate(cols):
        if h in ('smiles','smiles ' ,'smiles\t','smiles\n') or 'smiles' in h:
            smi_col = idx
        if 'catalog' in h or 'catalog id' in h or 'id' == h:
            id_col = idx
    if smi_col is None:
        smi_col = 0

    cnt = 0
    for row in reader:
        if len(row) <= smi_col:
            continue
        smi = row[smi_col].strip()
        if not smi:
            continue
        name = None
        
        if id_col is not None and len(row) > id_col and row[id_col].strip():
            name = safe_name(row[id_col].strip())
        else:
            if len(row) > 1 and row[1].strip():
                name = safe_name(row[1].strip())
            else:
                name = f"mol_{cnt:05d}"
        smi_file = os.path.join(out_dir, f"{name}.smi")

        if os.path.exists(smi_file):
            cnt += 1
            continue
        with open(smi_file, 'w') as g:
            g.write(smi + '\n')
        cnt += 1

print(f"Created {cnt} .smi files in {out_dir}")