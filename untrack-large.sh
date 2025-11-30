#!/usr/bin/env bash
# clean-large-tracked.sh
# Finds files > 49MB, then untracks them (so .gitignore can take effect)

set -euo pipefail

THRESHOLD="+49M"

echo "Looking for files larger than $THRESHOLD..."

# Use -print0 to handle whitespace, use xargs -0 for safety.
find . -type f -size "$THRESHOLD" -print0 \
  | while IFS= read -r -d '' f; do
      # Remove leading "./"
      clean_path="${f#./}"
      echo "Untracking: $clean_path"
      git rm --cached "$clean_path"
    done

echo "Now re-adding files according to .gitignore..."
git add .

echo -e "\nEnter commit message (leave empty to use default):"
read -p "Commit msg > " MSG

DEFAULT_MSG="clean: remove large files from tracking"

if [[ -z "$MSG" ]]; then
  git commit -m "$DEFAULT_MSG"
else
  git commit -m "$MSG"
fi
