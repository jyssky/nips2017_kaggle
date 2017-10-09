#!/bin/bash
#
# run_defense.sh is a script which executes the defense
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_defense.sh INPUT_DIR OUTPUT_FILE
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_FILE - file to store classification labels
#

INPUT_DIR=$1
OUTPUT_FILE=$2

if [[ "${OSTYPE}" == "darwin"* ]]; then
    TEMP_OUT_FILE_DIRECTORY="/private"$(mktemp -d)
else
    TEMP_OUT_FILE_DIRECTORY=$(mktemp -d)
fi

python defense.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="${OUTPUT_FILE}" \
  --checkpoint_path=adv_inception_v3.ckpt \
  --temp_output_file_dir="$TEMP_OUT_FILE_DIRECTORY"
