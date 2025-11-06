#!/bin/bash
# Script để activate CUDA và cuDNN cho realtime-transcript

# Activate conda env (thay v2t bằng env name của bạn nếu khác)
source ~/activate_cuda_env.sh
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
        echo "Added cuDNN lib path: $CUDNN_LIB_PATH"
    fi
    
    # Cũng thêm pytorch lib path (nếu có)
    TORCH_LIB_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
    if [ -d "$TORCH_LIB_PATH" ]; then
        export LD_LIBRARY_PATH="$TORCH_LIB_PATH:$LD_LIBRARY_PATH"
        echo "Added PyTorch lib path: $TORCH_LIB_PATH"
    fi
fi

echo "CUDA environment activated!"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"