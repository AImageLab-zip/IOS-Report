#!/bin/bash
# Setup script to configure environment for Point-BERT extensions

# Activate virtual environment
source venv/bin/activate

# Add PyTorch libraries to LD_LIBRARY_PATH
TORCH_LIB=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH

# Add CUDA libraries to LD_LIBRARY_PATH (from the compilation output, we can see CUDA is at this path)
CUDA_LIB="/homes/admin/spack/opt/spack/linux-ivybridge/cuda-12.6.3-cr3dswdcqnxbg772az4phx3g6qqmegwy/lib64"
if [ -d "$CUDA_LIB" ]; then
    export LD_LIBRARY_PATH=$CUDA_LIB:$LD_LIBRARY_PATH
fi

echo "Environment configured successfully"
echo "LD_LIBRARY_PATH includes PyTorch and CUDA libraries"
