#!/usr/bin/env bash
set -euo pipefail

TARGET="/work/grana_maxillo/IOS-DraftReport/_data/Dataset_FerraraDump_400_and_Bits2Bites"
SRC1="/work/grana_maxillo/IOS-DraftReport/_data/Dataset_FerraraDump_400"
SRC2="/work/grana_maxillo/IOS-DraftReport/_data/Dataset_Bits2Bites"

mkdir -p "$TARGET"

# Iterate over both source folders
for src in "$SRC1"/* "$SRC2"/*; do
  [ -e "$src" ] || continue
  name=$(basename "$src")
  if [ -e "$TARGET/$name" ]; then
    # Name collision: append the parent directory name (source folder) to disambiguate
    rel=$(basename "$(dirname "$src")")
    if [[ "$name" == *.* ]]; then
      base="${name%.*}"
      ext="${name##*.}"
      newname="${base}_${rel}.${ext}"
    else
      newname="${name}_${rel}"
    fi
    ln -s "$src" "$TARGET/$newname"
  else
    ln -s "$src" "$TARGET/$name"
  fi
done

echo "Created symlinks in $TARGET"
echo "Total items:" $(ls -1A "$TARGET" | wc -l)
ls -l "$TARGET" | head -n 20
