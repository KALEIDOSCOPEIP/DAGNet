# Differentiated Attention Guided Network over Hierarchical and Aggregated Features for Intelligent UAV Surveillance

Source code for testing the detection performance of DAGNet on infrared UAV targets. The codes are validated on a server with NVIDIA RTX 3090.

## Software installation

Basic environmental setups:

 - Python 3.9
 - CUDA 11.1
 - cnDNN 8.0.5

Python environmental setups:

 - lxml==4.6.3
 - matplotlib==3.4.1
 - mmcv==1.7.0
 - mmcv_full==1.6.0
 - numpy==1.20.1
 - opencv_python==4.5.4.60
 - torch==1.8.1+cu111
 - torchvision==0.9.1+cu111
 - tqdm==4.59.0

For Python environmental setups, run `pip install -r requirements.txt`.

## Weight file

The weight file can be downloaded from <a href='https://drive.google.com/file/d/1dU4M4tT83DGpYTxZi_NT9NUM4E3bK2MY/view?usp=sharing'>Google Drive</a> / <a href='https://pan.baidu.com/s/1WgAyeoX8geioCHGLczxsdw'>BaiduYun</a> (password: code).

The weight file should be placed in the ***weights*** directory, which is not present in the repository and is required to be created manually.

## Test images

The images for test purpose are given in the folder named `test_imgs`

## Testing

Simply run the following code in the console.
```Shell
python detect.py --weights ./weights/test.pth --image_dir ./test_imgs/ --output ./detection/ --cuda True
```
If you wanna run the script with CUDA device, set `--cuda True`, otherwise `--cuda False`.

By running the script, the detection results of the sample images will be saved in the `detection` directory that will appear after the finishing running the script. In each detection result image, the red box indicate the detected UAV target.