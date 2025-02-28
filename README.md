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


NOTE：Initially, we calculated Fm by averaging precision and recall, then using these averages to compute Fm. Thanks to feedback, we identified this bug and revised the approach to compute Fm for each image individually before averaging. We have updated the `./utils/saliency_metric.py` file to fix this issue. Additionally, we have updated the results on the DIS-VD dataset and included the HCE metric. The updated results are shown below:
<p align="center">
    <img src="https://github.com/user-attachments/assets/3b74e9ca-72b3-4847-93aa-f8f021607b22" alt="image" width="900"/>
</p>

Here are some of our visual results:
<p align="center">
    <img src="https://github.com/user-attachments/assets/ac1738c5-c866-47c2-93b0-bba93c1e1ec5" alt="image" width="900"/>
</p>


## I. Requiremets

1. Clone this repository
```
git clone git@github.com:qianyu-dlut/MVANet.git
cd MVANet
```

2.  Install packages

```
conda create -n mvanet python==3.7
conda activate mvanet
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html 
pip install -U openmim
mim install mmcv-full==1.3.17
pip install -r requirements.txt
```

## II. Training
1. Download the dataset [DIS5K](https://drive.google.com/file/d/1O1eIuXX1hlGsV7qx4eSkjH231q7G1by1/view?usp=sharing) and update `image_root` `gt_root` in `./train.py` (line 39-40).
1. Download the pretrained model at [Google Drive](https://drive.google.com/file/d/1-Zi_DtCT8oC2UAZpB3_XoFOIxIweIAyk/view?usp=sharing) and update the pretrained model path in `./model/SwinTransformer.py` (line 643)
2. Then, you can start training by simply running:
```
python train.py
```

## III. Testing
1. Update the data path in config file `./utils/config.py` (line 3~7)
2. Replace the existing path with the path to your saved model in `./predict.py` (line 14)

    You can also download our trained model at [Google Drive](https://drive.google.com/file/d/1_gabQXOF03MfXnf3EWDK1d_8wKiOemOv/view?usp=sharing).
3. Start predicting by:
```
python predict.py
```
4. Change the predicted map path in `./test.py` (line 19) and start testing:
```
python test.py
```

You can get our prediction maps  at [Google Drive](https://drive.google.com/file/d/1qN9mVNK9hfS_a1radFQ9QNYsAQo9FpYS/view?usp=sharing).

5. You can get the FPS performance by running:
```
python test_fps.py
```

## Contact
If you have any questions, please feel free to contact me(ms.yuqian AT mail DOT dlut DOT edu DOT cn).

## Citations
```
@inproceedings{MVANet,
  title={Multi-view Aggregation Network for Dichotomous Image Segmentation},
  author={Yu, Qian and Zhao, Xiaoqi and Pang, Youwei and Zhang, Lihe and Lu, Huchuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3921--3930},
  year={2024}
}

```
