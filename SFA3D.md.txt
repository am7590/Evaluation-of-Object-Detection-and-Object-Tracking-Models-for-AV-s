Create conda env

```
conda create -n myenv python=3.8
conda activvate myenv
```

Install pytorch for proper cuda driver support. The version of python-cuda may change depending on your system
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

```
pip install numpy==1.22.4
pip install easydict
pip install matplotlib
pip install tqdm
pip install wget
pip install tensorboard
pip install torchsummary
pip install opencv-python
pip install scikit-learn
```

Then, install from SFA3D's github page. Do not follow the installation files, instead use the above steps
https://github.com/maudzung/SFA3D

Follow the instructions on the page for running training and inference.
