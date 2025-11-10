#!/bin/bash
# Script để chạy server với CUDA/cuDNN đã được setup

cd "$(dirname "$0")"

# Activate conda env và setup CUDA paths
source ~/activate_cuda121.sh # you need to install cuda 12.1 first, and then activate it
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate v2t

# Set CUDA paths (nếu có CUDA local)
if [ -d "$HOME/cuda-12.1" ]; then
    export CUDA_HOME=$HOME/cuda-12.1
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi

# Thêm cuDNN libraries từ conda env vào LD_LIBRARY_PATH
if [ -n "$CONDA_PREFIX" ]; then
    CUDNN_LIB_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib"
    if [ -d "$CUDNN_LIB_PATH" ]; then
        export LD_LIBRARY_PATH="$CUDNN_LIB_PATH:$LD_LIBRARY_PATH"
    fi
    
    # Cũng thêm pytorch lib path (nếu có)
    TORCH_LIB_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
    if [ -d "$TORCH_LIB_PATH" ]; then
        export LD_LIBRARY_PATH="$TORCH_LIB_PATH:$LD_LIBRARY_PATH"
    fi
fi

echo "Starting server with CUDA support..."
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Chạy server
CUDA_VISIBLE_DEVICES=1 uvicorn main_sensevoice:app --reload --host 0.0.0.0 --port 8918