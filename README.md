# MVANet
The official repo of the CVPR 2024 paper, Multi-view Aggregation Network for Dichotomous Image Segmentation

## Introduction
Dichotomous Image Segmentation (DIS) has recently emerged towards high-precision object segmentation from high-resolution natural images. When designing an effective DIS model, the main challenge is how to balance the semantic dispersion of high-resolution targets in the small receptive field and the loss of high-precision details in the large receptive field. Existing methods rely on tedious multiple encoder-decoder streams and stages to gradually complete the global localization and local refinement. 

Human visual system captures regions of interest by observing them from multiple views. Inspired by it, we model DIS as a multi-view object perception problem and provide a parsimonious multi-view aggregation network (MVANet), which unifies the feature fusion of the distant view and close-up view into a single stream with one encoder-decoder structure. Specifically, we split the high-resolution input images from the original view into the distant view images with global information and close-up view images with local details. Thus, they can constitute a set of complementary multi-view low-resolution input patches.
![image](https://github.com/qianyu-dlut/MVANet/assets/73575386/2cff2cc2-ca24-469b-98ab-ed2585329609)

Moreover, two efficient transformer-based multi-view complementary localization and refinement modules (MCLM & MCRM) are proposed to jointly capturing the localization and restoring the boundary details of the targets. 
![image](https://github.com/qianyu-dlut/MVANet/assets/73575386/14c3e234-bdfe-49a5-a5ed-c82cc776d947)

We achieves state-of-the-art performance in terms of almost all metrics on the DIS benchmark dataset. 
![image](https://github.com/qianyu-dlut/MVANet/assets/73575386/6f3c0c1b-6cc2-4f0d-b563-7dc0c9050a52)

![image](https://github.com/qianyu-dlut/MVANet/assets/73575386/3c4443d8-fd6f-49f3-988d-45215bc1d8e6)


## To Do List
- Release our camere-ready paper on arxiv (done)
- Release our training code (to do)
- Release our model checkpoints (to do)
- Release our prediction maps (to do)


