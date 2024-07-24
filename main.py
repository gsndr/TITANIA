import tensorflow as tf
import os
import datetime

my_seed = 42

import random
import sys
import configparser
from Preprocessing import Preprocessing

random.seed(my_seed)

tf.random.set_seed(my_seed)
os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ['TF_DETERMINISTIC_OPS'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC'] = '0'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
os.environ['PYTHONHASHSEED'] = '1'
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
from tiler import Tiler, Merger


import HyperoptSDistillation



from sklearn.metrics import  confusion_matrix


if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


def _returnNameImage(m_path):
    name = m_path.split('_')[1]
    name = name.split('.')[0]
    return name

def _getTiler(shape, tile_shape,mode):
    tiler = Tiler(data_shape=shape,
                  tile_shape=tile_shape, mode=mode,
                  channel_dimension=None)
    return tiler

def main():
    dataset = sys.argv[1]
    config = configparser.ConfigParser()
    config.read('CONFIG.conf')
    dsConf = config[dataset]
    settings = config['setting']

    pathTrainI = dsConf.get('pathDatasetTrain')
    pathTestI = dsConf.get('pathDatasetTest')

    pathTrainM = dsConf.get('pathDatasetTrainM')
    pathTestM = dsConf.get('pathDatasetTestM')

    pathModel = dsConf.get('pathModels')
    shape = int(dsConf.get('shape'))
    ch = int(dsConf.get('channels'))
    tiles = int(dsConf.get('tiles'))
    tilesSize = int(dsConf.get('tilesSize'))

    resize = int(settings.get('RESIZE'))

    attention = int(settings.get('ATTENTION'))



    attack = int(dsConf.get('attack'))
    ch = int(dsConf.get('channels'))




    ds = dsConf.get('ds')

    #Define the name of the model according with the configuration selected
    name_model = pathModel + 'unet_resize_' + str(resize)
    

    rChannel = int(dsConf.get('resizeChannel'))

    if (int(settings.get('PREPROCESSING')) == 1):
       
        from PreprocessingImages import PreprocessingImages
        preprocessing=PreprocessingImages((tilesSize,tilesSize,ch), (tilesSize,tilesSize), rChannel, [12], tiles=1)

        preprocessing.images(pathTrainI, True)

        preprocessing.images(pathTestI, False)
        if (int(settings.get('PREPROCESSING_MASKS')) == 1):
            preprocessing.masks(pathTrainM,  ds)
            preprocessing.masks(pathTestM,  ds)

    if tiles:
        pathTrainImage = pathTrainI + 'Tiles/'
        pathTestImage = pathTestI + 'Tiles/'
        pathTrainMask = pathTrainM + 'Tiles/'
        pathTestMask = pathTestM + 'Tiles/'
    else:
        pathTrainImage = pathTrainI + 'Numpy/'
        pathTestImage = pathTestI + 'Numpy/'
        pathTrainMask = pathTrainM + 'Numpy/'
        pathTestMask = pathTestM + 'Numpy/'


    lst = os.listdir(pathTestMask)  # your directory path
    size_test = len(lst)

    if (int(settings.get('TRAIN_SELF_DISTILLATION')) == 1):
        import ast
        shallow = ast.literal_eval(settings.get('SHALLOW'))
        print("self")
        name_model = name_model + 'Self_Distillation'
        student = HyperoptSDistillation.trainNN(pathTrainImage, pathTestImage, pathTrainMask, pathTestMask,
                                                name_model + "_self", size_test, resize, (shape, shape, ch), shallow,
                                                attention=attention)


    if (int(settings.get('PREDICTION')) == 1):
        import numpy as np
        import ast
        shallow = ast.literal_eval(settings.get('SHALLOW'))

        
        name_model = pathModel + 'unet_resize_' + str(resize)+ 'Self_Distillation'+ "_self"
        shallow_list = str(shallow).replace('[', '').replace(']', '')
        name_model = name_model+f'{shallow_list}'
        if attention:
           name_model = name_model + '_attention'


        print(name_model)
       
        pathImagesSave = dsConf.get('pathSaveImages')  # pathto save images
        

        model = tf.keras.models.load_model(name_model + '_model.tf', compile=False)

        model.summary()

        import Utils

        col = ['tn', 'fp', 'fn', 'tp']
        import pandas as pd
        df = pd.DataFrame(columns=col)

        images_list = []
        mask_list = []
        import imageio

        for root, _, files in os.walk(pathTestI + 'Numpy/'):
            files.sort()
            # Here we sort to have the folder in alphabetical order
            for file in files:
                images_list.append(os.path.join(root, file))

        for root, _, files in os.walk(pathTestM + 'Numpy/'):
            files.sort()
            # Here we sort to have the folder in alphabetical order
            for file in files:
                mask_list.append(os.path.join(root, file))

        allPred = []
        allTrue = []
        prep = Preprocessing()
        

        for im_path, m_path in zip(images_list, mask_list):
           
            
            pred_image = []
            true_image = []
            image = np.load(im_path)
            mask = np.load(m_path)

            tiler = _getTiler(image.shape, (tilesSize, tilesSize, ch), mode='irregular')
            tilerMask = _getTiler(mask.shape, (tilesSize, tilesSize), mode='irregular')
            for tileImages, tileMasks in zip(tiler(image), tilerMask(mask)):
                img = tileImages[1]
                img = prep.resize_with_padding(img, size=((shape, shape, ch)))
                print(img.shape)
                m = tileMasks[1]
                
                pred, _ = Utils.predictionWithResizeSelfNoList(m, img, model,input_shape=(shape, shape, ch))

                

                #prediction = prep.reduce_padding(pred, m)
                #pred = ((prediction > 0.5) + 0).ravel()
                true = m.flatten()
                true_image.append(true)
                pred_image.append(pred)
             

            # Setup merging parameters
            imgAll = Utils.mergeImage(mask, pred_image)
            name=os.path.basename(os.path.normpath(m_path))
            name = _returnNameImage(name)
            imageio.imwrite(pathImagesSave + name + '.png', (imgAll * 255).astype(np.uint8))
            true_image = [arr.tolist() for arr in true_image]
            true_image = Utils.flattenList(true_image)
            pred_image = [arr.tolist() for arr in pred_image]
            pred_image = Utils.flattenList(pred_image)
            allTrue.append(true_image)
            allPred.append(pred_image)
            c = confusion_matrix(true_image, pred_image)
            if (len(c) > 1):
                tn, fp, fn, tp = c.ravel()
                r = [tn, fp, fn, tp]
            else:
                r = [c[0][0], 0, 0, 0]
            print(r)
            res = pd.DataFrame([r], columns=col)
            df = pd.concat([df, res], ignore_index=True)

        allTrue = Utils.flattenList(allTrue)
        allPred = Utils.flattenList(allPred)
        Utils.print_results(name_model, allTrue, allPred, model)
        print("Results reported in file: " + name_model + '_results.txt')
        df.to_csv(name_model + '_Singleresults.csv', index=False)


if __name__ == "__main__":
    main()
