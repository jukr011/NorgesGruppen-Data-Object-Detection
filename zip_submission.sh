#!/bin/bash
# Creates a submission zip for the competition.
# Run this AFTER training is complete and best.pt exists.
# Optionally also run build_reference_embeddings.py first to add re-ranking.
#
# Usage:  bash zip_submission.sh

set -e

if [ ! -f "best.pt" ]; then
  echo "ERROR: best.pt not found. Train the model first (python train.py ...)"
  exit 1
fi

OUTPUT="submission.zip"
rm -f "$OUTPUT"

# Core files (always included)
FILES="run.py requirements.txt best.pt"

# Optional re-ranking files (included if present)
[ -f "feature_extractor.pt" ]       && FILES="$FILES feature_extractor.pt"
[ -f "reference_embeddings.npy" ]   && FILES="$FILES reference_embeddings.npy"
[ -f "reference_labels.json" ]      && FILES="$FILES reference_labels.json"

echo "Creating $OUTPUT ..."
echo "  Including: $FILES"

zip "$OUTPUT" $FILES

SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo ""
echo "Done! $OUTPUT ($SIZE)"
echo ""

# Verify run.py is at root (not nested in a folder)
unzip -l "$OUTPUT" | grep "run.py" | head -3

echo ""
echo "Upload $OUTPUT at the competition submission page."
