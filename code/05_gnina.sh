#!/usr/bin/env bash
set -euo pipefail

# ==============================
# Usage:
#   sh gnina.sh --center 8vb5
#   sh gnina.sh --center aa1ac
# ==============================

# export LD_LIBRARY_PATH="/home/ssm-user/miniforge3/envs/docking/lib:$LD_LIBRARY_PATH"
export PATH="/home/ssm-user/apps/gnina/build/bin:$PATH"

BASE_DIR="/home/ssm-user/project/autodock"
RESULT_DIR="/home/ssm-user/project/gnina"
PROTEIN="$BASE_DIR/8vb5_prot.pdbqt"
LIGAND_DIRS=("$BASE_DIR/dot_potent_pdbqt" "$BASE_DIR/potent_pdbqt")
mkdir -p "$RESULT_DIR"

CENTER_REF="aa1ac"
while [[ $# -gt 0 ]]; do
    case $1 in
        --center)
            CENTER_REF="$2"
            shift 2
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            exit 1
            ;;
    esac
done

if [[ "$CENTER_REF" == "aa1ac" ]]; then
    CX=-1.128; CY=6.733; CZ=-18.115
elif [[ "$CENTER_REF" == "8vb5" ]]; then
    CX=-6.512; CY=1.343; CZ=-12.572
else
    echo "Error: No target center"
    exit 1
fi

echo ">>> Docking (center=$CENTER_REF, $CX $CY $CZ)"

FILES=()
for d in "${LIGAND_DIRS[@]}"; do
    FILES+=($(ls "$d"/*.pdbqt))
done

for f in "${FILES[@]}"; do
    LIGAND_NAME=$(basename "$f" .pdbqt)
    OUT_SDF="$RESULT_DIR/${LIGAND_NAME}_docked.sdf"
    LOG_FILE="$RESULT_DIR/${LIGAND_NAME}_gnina_out.log"

    echo "Start: $LIGAND_NAME"

    gnina -r "$PROTEIN" -l "$f" \
          --center_x $CX \
          --center_y $CY \
          --center_z $CZ \
          --size_x 20 \
          --size_y 20 \
          --size_z 20 \
          --exhaustiveness 10 \
          --cnn crossdock_default2018 \
          --cnn_scoring rescore \
          --seed 0 \
          -o "$OUT_SDF"
          # -o "$OUT_SDF" | tee "$LOG_FILE"
done