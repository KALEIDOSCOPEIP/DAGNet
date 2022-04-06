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

To run the testing source code script, please make sure the envionmental needs are met. For Python environmental setups, run `pip install -r requirements.txt`.

## Weight file

The weight file can be downloaded from <a href='https://drive.google.com/file/d/1dU4M4tT83DGpYTxZi_NT9NUM4E3bK2MY/view?usp=sharing'>Google Drive</a> / <a href='https://pan.baidu.com/s/1WgAyeoX8geioCHGLczxsdw'>BaiduYun</a> (password: code).

The weight file should be placed in the ***weights*** directory, which is not present in the repository and is required to be created manually.

## Testing

Simply run the following code in the console.
```Shell
python detect.py
```
By running the script, the detection results of the sample images will be saved in the `detection` directory that will appear after the finishing running the script. In each detection result image, the red box indicate the detected UAV target.
