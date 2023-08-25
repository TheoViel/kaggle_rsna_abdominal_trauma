# 2nd Place Solution to the Google Research - Identify Contrails to Reduce Global Warming Competition

**Authors :** [Theo Viel](https://github.com/TheoViel), [Iafoss](https://github.com/iafoss), [DrHB](https://github.com/DrHB)

## Introduction - Adapted from [Kaggle](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/430491)

This repository contains Theo's part of the solution. Only the efficientet_v2-s trained with external data is used in the final solution, and achieves private LB 0.700 in a 10-seeds ensemble. Adding the 2.5D models (5x convnext-v2-nano + 5x v2-s) boosts the ensemble to 0.706 private.

![](contrails_v2s.png)

### Details

This competition has two main challenges: (1) noisy labels and (2) pixel-level accuracy requirements. (1) If one checks the annotation from all annotators, a major disagreement is quite apparent. Fortunately, this label noise challenge may be addressed only by using soft labels (the average of all annotator labels) during training. Since the evaluation requires hard labels, the major model failure is predicting contrails near the decision boundary, and it cannot be avoided. (2) The thing that could really be addressed by the models is achieving the pixel-level accuracy of predictions. For such a task, a typical solution is upsampling the input of the model or replacing the final linear upsampling in segmentation models with conv or pixel shuffle upsampling. This modification gave us a substantial boost in initial experiments.

### Data

Our solution is based on false color images: band 14, and the difference between bands 12-14 and 14-15. We tried to consider more bands, but it was not helpful. The reason may be that annotators used the same input in their work. The input to the model was upscaled to 512 or 1024 with bicubic interpolation, while Efnet models were modified to have a stride of 1 in the first convolution and used a 256 input size. In our experiments, we realized that flip and 90 rotation augmentation are decreasing the model performance (though we didn't realize that the masks are shifted). Therefore, we trained the models with pixel-based augmentation + small angle rotation

### Model

The interesting thing about this competition is that the model is required to do both (1) tracking the global dependencies because contrails are quite elongated, and (2) capable of generating predictions with pixel-level accuracy because contrails are only several pixels thick (even mistakes in a single pixel may lead to a tremendous decrease of the model performance).

In the [organizer's paper](https://arxiv.org/pdf/2304.02122.pdf), it is suggested that consideration of image sequence is preferable in comparison to single frame models. Also, the instruction to annotators requires the contrails to be present in at least 2 frames. Therefore, we considered image sequences, like the 0th-5th frame or 3rd-6th frames. In our early experiments, it became apparent that 3d and video models are not expected to work for the considered data: the time delay between input frames is too huge, and the images shift too much. Therefore, the reasonable choice was using 2d models with temporal mixing at feature maps followed by standard Unet style upscaling. Consideration of high-resolution feature maps is also not meaningful when clouds shift by 30+ pixels between frames (we upscale the input). So we considered temporal mixing only at res/32 and res/16 feature maps. The best mixing strategy based on our experiments was using LSTM. Unfortunately, LSTM assumes spatial alignment of features, and we tried to use a transformer applied to flattened token sequence from all frames at res/32 and res/16 to do implicit image registration and contrails tracking, but it worked slightly worse. We also considered conv-based temporal mixing.

The model setup is schematically illustrated bellow. We process all input images with the backbone and then apply temporal mixing at res/16 feature scales. For other feature maps, we just pool the 5th frame output without mixing with others. This part is followed with vanilla U-Net decoder.

### Training

EfficientNet models were trained with AdamW. Typical training takes ~30 epochs, and 100 or 200 epochs when using external data. Code for training convnext models is also provided. We used BCE + dice Lovasz loss. The latter contribution is a modification of standard Lovasz (symmetric one with elu+1) to approximate dice instead of IoU. This component is using global statistics to approximate global dice used in the competition. Lovasz loss made our predictions less susceptible to the threshold selection making the maximum of dice wider. Given the nosy nature of CV and LB, this property is quite important for stable CV and avoiding shakeups at private LB, even if we saw only a minor improvement at CV when added Lovasz term.


## How to use the repository

### Prerequisites

- Clone the repository

- Download the data in the `input` folder:
  - [Competition data](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/data)

- (Optional) Download the pseudo-labeled masks and images from GOES16 in the `output` folder: 
  - [Images](https://www.kaggle.com/datasets/theoviel/contrails-goes16-img-1)
  - [Masks](https://www.kaggle.com/datasets/theoviel/contrails-goes16-mask-1)

Structure should be :
```
output
├── goes16_pl
│   ├── images
│   │   ├── 121_00_202312199999293.png
│   │   └── ...
│   └── masks
│       ├── 121_00_202312199999293.npy
│       └── ...
└── df_goes16_may.csv
```

- Setup the environment :
  - `pip install -r requirements.txt`

- We also provide trained model weights :
  - [2.5D Models](https://www.kaggle.com/datasets/theoviel/contrail-weights-v1)
  - [2D Models](https://www.kaggle.com/datasets/theoviel/contrail-weights-2d)


### Run The pipeline

#### Preparation

Preparation is done is the `notebooks/Preparation.ipynb` notebook. This will save the frame 4 in png using the false_color scheme in png for faster 2D training.

#### Training

- `bash train.sh` will train the 2D models. Downloading the external data is required.
- `bash train_end2end.sh` will train the 2D model and finetune it on 2.5D. 

#### Validation

Validation is done is the `notebooks/Validation.ipynb` notebook. Make sure to replace the `EXP_FOLDER` vairables with your models

#### Inference

Inference is done on Kaggle, notebook is [here](https://www.kaggle.com/code/theoviel/contrails-inference-comb).


### Code structure

If you wish to dive into the code, the repository naming should be straight-forward. Each function is documented.
The structure is the following :

```
src
├── data
│   ├── dataset.py              # Dataset class
│   ├── loader.py               # Dataloader
│   ├── preparation.py          # Data preparation
│   ├── shape_descript.py       # Shape descriptors aux target
│   └── transforms.py           # Augmentations
├── model_zoo 
│   ├── aspp.py                 # ASPP center block
│   ├── models.py               # Segmentation model wrapper
│   ├── transformer.py          # Transformer temporal mixer
│   └── unet.py                 # Customized U-Net
├── training                        
│   ├── losses.py               # Losses
│   ├── lovasz.py               # Lovasz loss
│   ├── main.py                 # k-fold and train function
│   ├── meter.py                # Segmentation meter
│   ├── mix.py                  # Cutmix and Mixup
│   ├── optim.py                # Optimizers
│   └── train.py                # Torch fit and eval functions
├── util
│   ├── logger.py               # Logging utils
│   ├── metrics.py              # Metrics for the competition
│   ├── plots.py                # Plotting utils
│   ├── rle.py                  # RLE encoding
│   └── torch.py                # Torch utils
├── inference_main.py           # Inference functions
├── main_end2end_convnext.py    # Pretrains and trains a convnext-nano 2.5D model
├── main_end2end_v2s.py         # Pretrains and trains a v2-s 2.5D model
├── main.py                     # Trains a v2-s 2D model on external data
└── params.py                   # Main parameters
``` 
