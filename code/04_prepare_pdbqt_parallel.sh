#!/usr/bin/env bash
set -euo pipefail

BASE=/home/ssm-user/project/autodock
SMI_DIR=$BASE/potent
OUT_DIR=$BASE/potent_pdbqt
TMP_DIR=$BASE/tmp_sdf
LOG_FAIL=$BASE/failures.log
JOBS=${JOBS:-$(nproc)}

mkdir -p "$OUT_DIR" "$TMP_DIR"
: > "$LOG_FAIL"

# check tools
if ! python3 -c "import rdkit" >/dev/null 2>&1; then
  echo "ERROR: RDKit is not available in python3. Activate environment with RDKit." >&2
  exit 1
fi

if ! command -v mk_prepare_ligand.py >/dev/null 2>&1; then
  echo "WARN: mk_prepare_ligand.py not found in PATH. Make sure it's accessible." >&2
  # you can exit here if it's required
  # exit 1
fi

process_one() {
  smi_path="$1"
  base=$(basename "$smi_path" .smi)
  sdf_path="$TMP_DIR/${base}.sdf"
  pdbqt_out="$OUT_DIR/${base}.pdbqt"

  # skip if already exists
  if [ -f "$pdbqt_out" ]; then
    echo "[skip] $base already exists."
    return 0
  fi

  # 1) SMILES -> SDF using RDKit (embedded python)
  python3 - <<PY
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
smi_file = r"${smi_path}"
sdf_file = r"${sdf_path}"
with open(smi_file,'r') as f:
    smi = f.read().strip()
mol = Chem.MolFromSmiles(smi)
if mol is None:
    print("FAILED: RDKit could not parse SMILES", file=sys.stderr)
    sys.exit(2)
mol = Chem.AddHs(mol)
# embed
res = AllChem.EmbedMolecule(mol,randomSeed=42)
if res != 0:
    # try a more robust embed
    res2 = AllChem.EmbedMolecule(mol,randomSeed=0,useRandomCoords=True)
    if res2 != 0:
        print("FAILED: Embed failed for", file=sys.stderr)
        # still try to write 2D? exit with failure
        sys.exit(3)
AllChem.UFFOptimizeMolecule(mol,maxIters=200)
w = Chem.SDWriter(sdf_file)
w.write(mol)
w.close()
print("OK")
PY

  rc=$?
  if [ $rc -ne 0 ]; then
    echo "$base : RDKit->SDF failed (rc=$rc)" >> "$LOG_FAIL"
    rm -f "$sdf_path"
    return 1
  fi

  # 2) SDF -> PDBQT via mk_prepare_ligand.py (or relevant tool)
  # Adjust mk_prepare_ligand.py args as needed for your environment.
  if command -v mk_prepare_ligand.py >/dev/null 2>&1; then
    mk_prepare_ligand.py -i "$sdf_path" -o "$pdbqt_out"
    rc2=$?
  else
    # fallback: try AutoDockTools prepare_ligand4.py if available
    if command -v prepare_ligand4.py >/dev/null 2>&1; then
      # get pdbqt by using prepare_ligand4.py
      # note: prepare_ligand4.py usually reads mol2/pdbqt options; adjust if needed
      prepare_ligand4.py -l "$sdf_path" -o "$pdbqt_out" || rc2=$?
    else
      echo "NO_CONVERTER: no mk_prepare_ligand.py or prepare_ligand4.py available" >> "$LOG_FAIL"
      rc2=127
    fi
  fi

  if [ "${rc2:-0}" -ne 0 ]; then
    echo "$base : SDF->PDBQT conversion failed (rc=${rc2:-0})" >> "$LOG_FAIL"
    rm -f "$sdf_path" "$pdbqt_out"
    return 1
  fi

  # cleanup tmp sdf
  rm -f "$sdf_path"
  echo "$base : OK"
  return 0
}

export -f process_one
export TMP_DIR OUT_DIR LOG_FAIL

# build list of smi
find "$SMI_DIR" -maxdepth 1 -type f -name "*.smi" | sort > /tmp/smi_list.txt

# If GNU parallel is available, use it; otherwise fallback to xargs -P
if command -v parallel >/dev/null 2>&1; then
  cat /tmp/smi_list.txt | parallel -j "$JOBS" --halt soon,fail=1 process_one {}
else
  cat /tmp/smi_list.txt | xargs -n1 -P "$JOBS" -I {} bash -c 'process_one "$@"' _ {}
fi

echo "Done. Failures (if any) in $LOG_FAIL"