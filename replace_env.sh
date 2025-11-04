#!/bin/bash

INPUT_FILE="docker-compose.prod.yml"
TEMP_FILE="${INPUT_FILE}.tmp"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: $INPUT_FILE not found"
    exit 1
fi

# Find all $VAR_NAME patterns and check if they're set
missing_vars=()
while IFS= read -r line; do
    if [[ $line =~ \$([A-Z_]+) ]]; then
        var_name="${BASH_REMATCH[1]}"
        if [ -z "${!var_name}" ]; then
            if [[ ! " ${missing_vars[@]} " =~ " ${var_name} " ]]; then
                missing_vars+=("$var_name")
            fi
        fi
    fi
done < "$INPUT_FILE"

# Exit if any variables are missing
if [ ${#missing_vars[@]} -gt 0 ]; then
    echo "Error: The following environment variables are not set:"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    exit 1
fi

# Replace variables with actual values
while IFS= read -r line; do
    output_line="$line"
    while [[ $output_line =~ \$([A-Z_]+) ]]; do
        var_name="${BASH_REMATCH[1]}"
        var_value="${!var_name}"
        output_line="${output_line//\$$var_name/$var_value}"
    done
    echo "$output_line"
done < "$INPUT_FILE" > "$TEMP_FILE"

mv "$TEMP_FILE" "$INPUT_FILE"

echo "✓ Replaced environment variables in $INPUT_FILE"

