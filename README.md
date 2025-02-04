#  self-DisTIlled uneT for Sentinel-2 imAge semaNtIc segmentAtion  (TITANIA)


The repository contains code referred to the work:

_Giuseppina Andresini, Annalisa Appice,  Donato Malerba_

[A Deep Semantic Segmentation Approach to Map Forest Tree Dieback in Sentinel-2 Data](https://ieeexplore.ieee.org/document/10680607) 

Please cite our work if you find it useful for your research and work.
```
@ARTICLE{10680607,
  author={Andresini, Giuseppina and Appice, Annalisa and Malerba, Donato},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={A Deep Semantic Segmentation Approach to Map Forest Tree Dieback in Sentinel-2 Data}, 
  year={2024},
  volume={17},
  number={},
  pages={17075-17086},
  keywords={Forestry;Vegetation;Tensors;Remote sensing;Titanium dioxide;Monitoring;Accuracy;Attention;forest tree die-back monitoring;forest wildfires monitoring;insect outbreak monitoring;self-distillation;semantic segmentation;Sentinel-2 image processing},
  doi={10.1109/JSTARS.2024.3460981}}

```


## Code requirements
The code relies on the following python3.9+ libs.
Packages needed are:
* Tensorflow 2.15
* Numpy 1.26.3
* Hyperopt 0.2.7
* Keras 2.15.0
* Scikit-learn 1.3.0


## How to use
Repository contains scripts of all experiments included in the paper:
* __main.py__ : script to run  TITANIA 
To run the code the command is main.py NameOfDataset (es FRANCE10, FIRES)
The details of dataset should be included in the file ___CONFIG.conf__
For example:
```python
[FRANCE10]
pathModels = Models/BB/October/
pathDatasetTrain = ../DS/SWIFTT/SRI_FRANCE/October/Train/
pathDatasetTest = ../DS/SWIFTT/SRI_FRANCE/October/Test/
pathDatasetTrainM = ../DS/SWIFTT/SRI_FRANCE/Masks/Train/
pathDatasetTestM = ../DS/SWIFTT/SRI_FRANCE/Masks/Test/
#number of files in the folder tiles of the test
sizetest=543
#number of channels
channels=12
#shape of the tiled image
tilesSize=32
#if 1 create tiles
tiles=1
```

#Data
The dataset should be split into four folders: images of the test, images of the train, masks of the test, and masks of the train. 
Each folder contains: 
* Folder tiff containing the image in tiff format
* Folder numpy that is used to store the numpy version of the tiff files
* Folder tiles containing the tile used for computation
See an example in the folder DS associated to this repository
 
## Replicate the experiments

To replicate experiments reported in the work, you can use models and datasets stored in homonym folders.
Global variables are stored in __CONFIG.conf__  file 


```python
[setting]
#if the images are of different size performs the padding
RESIZE=1
#to use the channel attention
ATTENTION=1
#to train self distillation
TRAIN_SELF_DISTILLATION=1
#for prediction
PREDICTION=1
#to preprocess images (e.g., creation of numpy, tiles)
PREPROCESSING=1
#to preprocess masks
PREPROCESSING_MASKS=1
#select the shallow models to be used for the creation of the model
SHALLOW=[5,4,3]
```

## Download datasets

[Bark Beetle in France](https://drive.google.com/drive/folders/11JPIK6cfgXdMY0PG4YHh6z8fuswBW3JO?usp=sharing)

[FIRES](https://drive.google.com/drive/folders/11bdq4pyRjLD37QV9o7dyvgcHt79xEztk?usp=sharing)

## Download trained models 

[Models](https://drive.google.com/drive/folders/1XntFkX4kzJkqDUu6AO2144bojXlzS6Vi?usp=sharing)








