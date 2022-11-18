# SVR

Code for paper "Saliency-aware Stereoscopic Video Retargeting"


![alt text]([http://url/to/img.png](https://drive.google.com/file/d/1XFHFNJJ0O6AdfvgiqIeXnZE6QRGJeaiJ/view?usp=share_link))

# Training

## Download the datasets
First fownload the KITTI stereo video 2015 and 2012 datasets from the following link:

https://www.cvlibs.net/datasets/kitti/eval_stereo.php


## Download the weights of CoSD

Download the weights of the saliency detection method from the following like:

https://github.com/suyukun666/UFO

And put them in the "models/" folder

For calculating the perceptual loss, clone the following repository and put it in the "PerceptualSimilarity/" directory:

https://github.com/SteffenCzolbe/PerceptualSimilarity



Run the following command:
sudo python3 main.py


