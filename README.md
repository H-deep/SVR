# SVR

Code for paper "Saliency-aware Stereoscopic Video Retargeting"


### About

**SVR (Stereo Vision Reconstruction)** is a transformer-based framework for stereo image understanding, integrating multi-scale attention and intra-MLP reasoning for precise disparity and mask prediction.  
Originally published in a **CVPR-grade study**, this project extends classical stereo super-resolution and segmentation methods toward **edge-deployable AI** via ONNX and TensorRT optimization.



![alt text](https://github.com/z65451/SVR/blob/main/model.jpg)

![alt text](https://github.com/z65451/SVR/blob/main/result.png)

# Training

First install the requirements:

```
pip3 install requirements
```


## Download the datasets
First fownload the KITTI stereo video 2015 and 2012 datasets from the following link:

https://www.cvlibs.net/datasets/kitti/eval_stereo.php

Unzip and put the datasets in the "datasets/" directory.

## Download the weights of CoSD [1]

Download the weights of the saliency detection method from the following like:

https://github.com/suyukun666/UFO

Put them in the "models/" directory.

For calculating the perceptual loss [2], clone the following repository and put it in the "PerceptualSimilarity/" directory:

https://github.com/SteffenCzolbe/PerceptualSimilarity

## Start training

Run the following command to start the training:

```
python3 main.py --train_or_test train
```

The trained model will be saved in the "models/" directory.


# Testing
Download the pre-trained model from th following link as put it in "models/" directory:

https://drive.google.com/file/d/11mlx1PRh-oFzOTLABAkySk7_DGLvEmyw/view?usp=share_link

Run the following command to start testing:

```
python3 main.py --train_or_test test
```


## ⚡️ Edge Deployment (ONNX / TensorRT)

This repository now includes an **edge-ready export pipeline** for the **SVR** model — enabling deployment of the stereo transformer-based segmentation network on **NVIDIA Jetson**, **AGX Orin**, or **RTX Edge devices**.  
The exported model focuses on the lightweight **mask prediction branch (`build_model2`)**, optimized for embedded inference while preserving the transformer reasoning structure.

---

### 🔹  Export to ONNX
Convert the trained PyTorch checkpoint (`.pth`) into a portable **ONNX** model:

```bash
python export_onnx_svr.py \
  --ckpt models/UFO_best_ft.pth \
  --onnx outputs/svr/model_static.onnx \
  --img 224 224 \
  --opset 14 --device cuda
````

✅ **Output:** `outputs/svr/model_static.onnx`
**Input:** `(B, 3, 224, 224)` **Output:** `(B, 1, 224, 224)` (binary mask)

The exporter automatically handles `group_size=5` tiling internally to ensure consistency with the original model behavior.

---

### 🔹  Build TensorRT Engine (FP16 / INT8)

Convert the ONNX model into an optimized **TensorRT engine** for high-performance deployment:

**FP16 build**

```bash
python build_trt_svr.py \
  --onnx outputs/svr/model_static.onnx \
  --engine outputs/svr/model_fp16.engine \
  --fp16 \
  --min_H 224 --opt_H 224 --max_H 224 \
  --min_W 224 --opt_W 224 --max_W 224
```

** INT8 build (with calibration data):**

```bash
python build_trt_svr.py \
  --onnx outputs/svr/model_static.onnx \
  --engine outputs/svr/model_int8.engine \
  --int8 --calib_dir calib/ \
  --opt_H 224 --opt_W 224
```

> 💡 For INT8 calibration, save batches of `.npy` arrays (shape `(B, 3, H, W)`) inside a folder named `calib/`.

---

### 🔹 Edge Inference

The generated `.engine` file can be directly used for real-time inference with TensorRT in Python or C++, enabling:


### 🔹 Highlights

* 🧠 Transformer-based stereo mask prediction
* ⚙️ ONNX + TensorRT export for embedded deployment
* 🚀 FP16 / INT8 inference acceleration
* 🧩 Clean, single-input deployment pipeline


---


# References

[1] Yukun Su, Jingliang Deng, Ruizhou Sun, Guosheng Lin,
and Qingyao Wu. A unified transformer framework for
group-based segmentation: Co-segmentation, co-saliency
detection and video salient object detection. arXiv preprint
arXiv:2203.04708, 2022. 3

[2] Czolbe, Steffen, et al. "A loss function for generative neural networks based on watson’s perceptual model." Advances in Neural Information Processing Systems 33 (2020): 2051-2061.
