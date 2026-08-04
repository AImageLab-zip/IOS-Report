#!/bin/bash

# Script to add prefixes to each line in the generated captions
# Target directory: /work/grana_maxillo/CVPR2026/output/google_medgemma-27b-it

INPUT_DIR="/work/grana_maxillo/CVPR2026/output/google_medgemma-27b-it"

# Define the prefixes for each line
prefixes=(
    "Overbite: "
    "Crowding: "
    "Molar occlusion - Right side: "
    "Molar occlusion - Left side: "
    "Canine occlusion - Right side: "
    "Canine occlusion - Left side: "
    "Curve of Spee: "
    "Curve of Wilson: "
    "Midlines: "
    "Transverse relationship: "
    "Missing teeth: "
)

# Process each .txt file in the directory
for file in "$INPUT_DIR"/*.txt; do
    if [ -f "$file" ]; then
        echo "Processing: $(basename "$file")"
        
        # Create a temporary file
        temp_file=$(mktemp)
        
        # Read the file line by line and add prefixes
        line_number=0
        while IFS= read -r line || [ -n "$line" ]; do
            if [ $line_number -lt ${#prefixes[@]} ]; then
                echo "${prefixes[$line_number]}$line" >> "$temp_file"
            else
                # If there are more lines than prefixes, just copy them as-is
                echo "$line" >> "$temp_file"
            fi
            ((line_number++))
        done < "$file"
        
        # Replace the original file with the modified content
        mv "$temp_file" "$file"
    fi
done

echo "Done! Processed all files in $INPUT_DIR"
