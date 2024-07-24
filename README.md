#  (self-Dis{\sf TI}lled une{\sf T} for Sentinel-2 im{\sf A}ge sema{\sf N}tIc segment{\sf A}tion)  (TITANIA)


The repository contains code refered to the work:

_Giuseppina Andresini, Annalisa Appice,  Donato Malerba _

[A Deep Semantic Segmentation Approach to Map Forest Tree Dieback in Sentinel-2 Data]() 

Please cite our work if you find it useful for your research and work.
```


```



## Code requirements
The code relies on the following python3.7+ libs.
Packages needed are:
* Tensorflow 2.4.0
* Pandas 1.3.2
* Numpy 1.19.5
* Matplotlib 3.4.2
* Hyperopt 0.2.5
* Keras 2.4.3
* Scikit-learn 0.24.2
* [Visual-attention-tf 1.2](https://pypi.org/project/visual-attention-tf/)

## How to use
Repository contains scripts of all experiments included in the paper:
* __main.py__ : script to run  TITANIA 
To run the code the command is main.py NameOfDataset (es FRANCE10, FIRES)
  

 
## Replicate the experiments

To replicate experiments reported in the work, you can use models and datasets stored in homonym folders.
Global variables are stored in __CONFIG.conf__  file 


```python
[setting]
ATTACK_LABEL = 1
#se 1 viene effettuato il preprocessing del dataset
RESIZE=1
ATTENTION=1
TRAIN_SELF_DISTILLATION=1
PREDICTION=1
PREPROCESSING=1
PREPROCESSING_MASKS=1
SHALLOW=[5,4,3]
```

## Download datasets

[All datasets]()








