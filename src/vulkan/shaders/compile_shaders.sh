#!/bin/bash

# Script to compile GLSL shaders to SPIR-V

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_DIR="$SCRIPT_DIR/spirv"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find all .comp files and compile them
for shader in "$SCRIPT_DIR"/*.comp; do
    if [ -f "$shader" ]; then
        filename=$(basename "$shader" .comp)
        echo "Compiling $filename.comp..."
        glslangValidator -V "$shader" -o "$OUTPUT_DIR/$filename.spv"
        if [ $? -ne 0 ]; then
            echo "Error compiling $filename.comp"
            exit 1
        fi
    fi
done

echo "All shaders compiled successfully!"
