Source code for testing the detection performance of DAGNet on infrared UAV targets. The codes are validated on a server with NVIDIA RTX 3090.

## Software installation

Basic environmental setups:

 - Python 3.9
 - CUDA 11.1
 - cnDNN 8.0.5

Python environmental setups:

 - matplotlib==3.4.1
 - numpy==1.19.4
 - opencv_python==4.5.1.48
 - torch==1.8.1+cu111
 - torchvision==0.9.1+cu111

For Python environmental setups, run `pip install -r requirements.txt`.

**❗❗ For the deformable convolution environmental requirements: if your platform environment satisfies the above requirements perfectly, the testing script will automatically include the existing deformable convolution cython file we provide within folder `models/deform/`. By this, you can directly run the testing script as specified in the following Testing section. Otherwise, deformable convolution environment is required to be generated mannually due to environment discrepancies. You need to run the following code in the console, or package import error will occur:**

```Shell
cd ./models/deform/original/
make.sh
```

**If the setup process goes well, there should be one newly generated cython .so file at `models/deform/` folder that matches the python version of your environment. Any errors reported from gcc/g++ during this compiling procedure could be related to your gcc/g++ configuration mismatch.**

## Weight file

The weight file can be downloaded from <a href='https://drive.google.com/file/d/1dU4M4tT83DGpYTxZi_NT9NUM4E3bK2MY/view?usp=sharing'>Google Drive</a> / <a href='https://pan.baidu.com/s/1WgAyeoX8geioCHGLczxsdw'>BaiduYun</a> (password: code).

The weight file should be placed in the ***weights*** directory, which is not present in the repository and is required to be created manually.

## Testing

Simply run the following code in the console.
```Shell
python detect.py
```
By running the script, the detection results of the sample images will be saved in the `detection` directory that will appear after the finishing running the script. In each detection result image, the red box indicate the detected UAV target.
