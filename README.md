# SVR

Code for paper "Saliency-aware Stereoscopic Video Retargeting"


![alt text](https://github.com/z65451/SVR/blob/main/model.jpg)

![alt text](https://github.com/z65451/SVR/blob/main/result.png)

# Training

## Download the datasets
First fownload the KITTI stereo video 2015 and 2012 datasets from the following link:

https://www.cvlibs.net/datasets/kitti/eval_stereo.php

Unzip and put the datasets in the "datasets/" directory.

## Download the weights of CoSD

Download the weights of the saliency detection method from the following like:

https://github.com/suyukun666/UFO

Put them in the "models/" directory.

For calculating the perceptual loss, clone the following repository and put it in the "PerceptualSimilarity/" directory:

https://github.com/SteffenCzolbe/PerceptualSimilarity

## Start training

Run the following command to start the training:

```
sudo python3 main.py
```

The trained model will be saved in the "models/" directory.



data

├── cats_vs_dogs

│   ├── test_cat_dog

│   ├── testing

│   │   ├── cats

│   │   └── dogs

│   └── training

│       ├── cats

│       └── dogs

└── PetImages

    ├── Cat
    
    └── Dog




