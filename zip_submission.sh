#!/bin/bash
# Creates a submission zip for the competition.
# Run this AFTER training is complete and best.pt exists.
#
# Usage:  bash zip_submission.sh

set -e

if [ ! -f "best.pt" ]; then
  echo "ERROR: best.pt not found. Train the model first (python train.py ...)"
  exit 1
fi

OUTPUT="submission.zip"

echo "Creating $OUTPUT ..."
zip -r "$OUTPUT" \
  run.py \
  requirements.txt \
  best.pt

SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo "Done! $OUTPUT ($SIZE)"
echo ""
echo "Upload $OUTPUT at the competition submission page."
