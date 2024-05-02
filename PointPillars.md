# PointPillars Setup Instructions

Tested on reyne.cs.rit.edu.

## Create conda environment
```
conda create -n cudaEnv python=3.8
```

## Activate environment
```
conda activate cudaEnv
```

## Create conda environment
```
conda create -n cudaEnv python=3.8
```

## Install CUDA Toolkit
```
conda install cudatoolkit=11.3 -c conda-forge
```

## Install PyTorch
We could not get the correct versions downloaded via cuda.
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

```

## Export Environment Variables
Find your nvcc path with ```which nvcc```.

Set your environment variables as needed:
```
export PATH=/path/bin:$PATH
export CUDA_HOME=/path
```


## Verify installation
```
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version used by PyTorch: {torch.version.cuda}")
print(f"Is CUDA available: {torch.cuda.is_available()}")
```

## Follow Instructions on the PointPillars repo:
[PointPillars](https://github.com/zhulf0804/PointPillars)
