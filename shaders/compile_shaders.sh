#!/bin/bash

# Compile GLSL compute shaders to SPIR-V
# Requires glslc from Vulkan SDK

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLSL_DIR="$SCRIPT_DIR/glsl"
SPIRV_DIR="$SCRIPT_DIR/spirv"

# Create SPIR-V output directory
mkdir -p "$SPIRV_DIR"

echo "Compiling GLSL compute shaders to SPIR-V..."

# Check if glslc is available
if ! command -v glslc &> /dev/null; then
    echo "Error: glslc not found. Please install the Vulkan SDK."
    echo "Download from: https://vulkan.lunarg.com/"
    exit 1
fi

# Compile each shader
for shader in "$GLSL_DIR"/*.comp; do
    if [ -f "$shader" ]; then
        filename=$(basename "$shader")
        output="$SPIRV_DIR/${filename%.comp}.spv"

        echo "  Compiling $filename -> ${filename%.comp}.spv"
        glslc "$shader" -o "$output"

        if [ $? -eq 0 ]; then
            echo "    ✓ Success"
        else
            echo "    ✗ Failed"
            exit 1
        fi
    fi
done

echo ""
echo "All shaders compiled successfully!"
echo "Output directory: $SPIRV_DIR"
