#!/usr/bin/env bash
INPUT_FOLDER="${1}"

if [ ! -d "${INPUT_FOLDER}" ]; then
    echo "input folder not found: ${INPUT_FOLDER}"
    exit 1
fi

python scripts/download_models_flownet2.py

set -euo pipefail

python inference.py "${INPUT_FOLDER}" "${@:2}"

