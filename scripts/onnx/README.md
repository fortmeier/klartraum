# ONNX Network Examples Setup
Please create an environment for ONNX generation of onnx models.
Then check out the respective notebooks.

## Create Conda Environment

```bash
conda create -n klartraum-onnx python=3.12 ipykernel -y
conda activate klartraum-onnx
```

## Install ONNX with GPU Support

```bash
# Install CUDA toolkit (if not already installed)
conda install cudatoolkit cudnn -c conda-forge -y

# Install ONNX Runtime GPU

# pip install onnxruntime-gpu


pip install onnx

# Install Transformers with GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install transformers
pip install diffusers
pip install matplotlib
```
```

