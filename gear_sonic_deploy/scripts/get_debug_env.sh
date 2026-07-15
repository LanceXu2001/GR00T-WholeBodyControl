#!/bin/bash
# Helper script to get environment variables for debugging
# Usage: source scripts/get_debug_env.sh
# Then use the exported variables in your debugger

# Source the main setup script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup_env.sh"

# Export key variables for debugging
echo "=== Environment Variables for Debugging ==="
echo ""
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""
echo "TensorRT_ROOT=${TensorRT_ROOT:-not set}"
echo "CUDAToolkit_ROOT=${CUDAToolkit_ROOT:-not set}"
echo "CUDA_HOME=${CUDA_HOME:-not set}"
echo "HAS_ROS2=${HAS_ROS2:-0}"
echo ""
echo "=== Copy these to launch.json environment section ==="
echo ""
echo "To use in VSCode launch.json, add to environment array:"
echo ""
echo '{'
echo '    "name": "LD_LIBRARY_PATH",'
echo "    \"value\": \"$LD_LIBRARY_PATH\""
echo '},'
if [ -n "$TensorRT_ROOT" ]; then
    echo '{'
    echo '    "name": "TensorRT_ROOT",'
    echo "    \"value\": \"$TensorRT_ROOT\""
    echo '},'
fi
if [ -n "$CUDAToolkit_ROOT" ]; then
    echo '{'
    echo '    "name": "CUDAToolkit_ROOT",'
    echo "    \"value\": \"$CUDAToolkit_ROOT\""
    echo '},'
fi
if [ -n "$CUDA_HOME" ]; then
    echo '{'
    echo '    "name": "CUDA_HOME",'
    echo "    \"value\": \"$CUDA_HOME\""
    echo '},'
fi
