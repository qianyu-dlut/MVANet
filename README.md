# MVANet
The official repo of the CVPR 2024 paper (Highlight), [Multi-view Aggregation Network for Dichotomous Image Segmentation](https://arxiv.org/abs/2404.07445)

 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-view-aggregation-network-for/dichotomous-image-segmentation-on-dis-te1)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-te1?p=multi-view-aggregation-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-view-aggregation-network-for/dichotomous-image-segmentation-on-dis-te2)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-te2?p=multi-view-aggregation-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-view-aggregation-network-for/dichotomous-image-segmentation-on-dis-te3)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-te3?p=multi-view-aggregation-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-view-aggregation-network-for/dichotomous-image-segmentation-on-dis-te4)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-te4?p=multi-view-aggregation-network-for) 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-view-aggregation-network-for/dichotomous-image-segmentation-on-dis-vd)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-vd?p=multi-view-aggregation-network-for)
## Introduction
Dichotomous Image Segmentation (DIS) has recently emerged towards high-precision object segmentation from high-resolution natural images. When designing an effective DIS model, the main challenge is how to balance the semantic dispersion of high-resolution targets in the small receptive field and the loss of high-precision details in the large receptive field. Existing methods rely on tedious multiple encoder-decoder streams and stages to gradually complete the global localization and local refinement. 

Human visual system captures regions of interest by observing them from multiple views. Inspired by it, we model DIS as a multi-view object perception problem and provide a parsimonious multi-view aggregation network (MVANet), which unifies the feature fusion of the distant view and close-up view into a single stream with one encoder-decoder structure. Specifically, we split the high-resolution input images from the original view into the distant view images with global information and close-up view images with local details. Thus, they can constitute a set of complementary multi-view low-resolution input patches.
<p align="center">
    <img src="https://github.com/qianyu-dlut/MVANet/assets/73575386/2cff2cc2-ca24-469b-98ab-ed2585329609" alt="image" width="900"/>
</p>

Moreover, two efficient transformer-based multi-view complementary localization and refinement modules (MCLM & MCRM) are proposed to jointly capturing the localization and restoring the boundary details of the targets. 
<p align="center">
    <img src="https://github.com/qianyu-dlut/MVANet/assets/73575386/14c3e234-bdfe-49a5-a5ed-c82cc776d947" alt="image" width="900"/>
</p>


We achieves state-of-the-art performance in terms of almost all metrics on the DIS benchmark dataset. 
<p align="center">
    <img src="https://github.com/qianyu-dlut/MVANet/assets/73575386/6f3c0c1b-6cc2-4f0d-b563-7dc0c9050a52" alt="image" width="900"/>
</p>

We have optimized the code and achieved an enhanced FPS performance, reaching 15.2.
<p align="center">
    <img src="https://github.com/qianyu-dlut/MVANet/assets/73575386/4de86a52-5b55-4095-9a1f-afda40ce7f7a" alt="image" width="500"/>
</p>

Here are some of our visual results:
<p align="center">
    <img src="https://github.com/qianyu-dlut/MVANet/assets/73575386/3c4443d8-fd6f-49f3-988d-45215bc1d8e6" alt="image" width="900"/>
</p>


## I. Requiremets
+ python==3.7
+ torch==1.10.0
+ torchvision==0.11.0
+ mmcv-full==1.3.17
+ mmdet==2.17.0
+ mmengine==0.8.1
+ mmsegmentation==0.19.0
+ numpy
+ ttach
+ einops
+ timm
+ scipy

## II. Training
1. Download the pretrained model at [Google Drive](https://drive.google.com/file/d/1-Zi_DtCT8oC2UAZpB3_XoFOIxIweIAyk/view?usp=sharing).
2. Then, you can start training by simply run:
```
python train.py
```

## III. Testing
1. Update the data path in config file `./utils/config.py` (line 4~8)
2. Replace the existing path with the path to your saved model in `./predict.py` (line 14)

    You can also download our trained model at [Google Drive](https://drive.google.com/file/d/1_gabQXOF03MfXnf3EWDK1d_8wKiOemOv/view?usp=sharing).
3. Start predicting by:
```
python predict.py
```
4. Change the predicted map path in `./test.py` (line 17) and start testing:
```
python test.py
```

You can get our prediction maps  at [Google Drive](https://drive.google.com/file/d/1z21OMJ0Zl7JYKFxqR3P2YJTT3zay8doq/view?usp=sharing).
## To Do List
- Release our camere-ready paper on arxiv (done)
- Release our training code (done)
- Release our model checkpoints (done)
- Release our prediction maps (done)

## Citations
```
@article{yu2024multi,
  title={Multi-view Aggregation Network for Dichotomous Image Segmentation},
  author={Yu, Qian and Zhao, Xiaoqi and Pang, Youwei and Zhang, Lihe and Lu, Huchuan},
  journal={arXiv preprint arXiv:2404.07445},
  year={2024}
}
```
