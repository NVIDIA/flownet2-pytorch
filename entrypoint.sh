#!/usr/bin/env bash
INPUT_FOLDER="${1}"

if [ ! -d "$INPUT_FOLDER" ]; then
    echo "input folder not found: $INPUT_FOLDER"
    exit 1
fi

OUTPUT_FOLDER="${2:-$INPUT_FOLDER/feature}"
DATA_FOLDER="${INPUT_FOLDER}"/data

python scripts/download_models_flownet2.py

SUBDIRECTORIES="$(ls ${DATA_FOLDER})"

set -euo pipefail
for SUBDIRECTORY in ${SUBDIRECTORIES}; do
    NEWDIRECTORY="${SUBDIRECTORY}_feature-flow"
    INPUT="${DATA_FOLDER}"/"${SUBDIRECTORY}"
    OUTPUT="${OUTPUT_FOLDER}"/"${NEWDIRECTORY}"
    mkdir -p "${OUTPUT}"

    python main.py \
        --inference \
        --model FlowNet2 \
        --save_flow \
        --inference_dataset ImagesFromFolder \
        --inference_dataset_root "${INPUT}" \
        --resume checkpoints/FlowNet2_checkpoint.pth.tar \
        --save "${OUTPUT}"
done

