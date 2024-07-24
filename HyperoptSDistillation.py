import numpy as np
from keras import Model, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Dense, Flatten, Lambda, Dropout, Conv2DTranspose
from keras.optimizers import Adam

import SatelliteSelfUnet
import os
import csv

import losses
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)



my_seed = 12
np.random.seed(my_seed)
import random

random.seed(my_seed)

import tensorflow as tf

tf.random.set_seed(12)
from losses import *
import numpy as np
from hyperopt import STATUS_OK
from hyperopt import tpe, hp, Trials, fmin
from keras import backend as K
from keras.utils import to_categorical
from sklearn.metrics import balanced_accuracy_score

from sklearn.model_selection import train_test_split

import time



from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import image_generator
import Utils
import math

XGlobal = []
YGlobal = []

XTestGlobal = []
YTestGlobal = []

SavedParameters = []
Mode = ""
best_val_acc = 0
best_val_loss = np.inf
paramsGlobal = 0
best_model = None
teacherModel = 0
trainable = 0
iteration=1





def trainNN(trainImage,testingImage, trainMask, testMask, name, size_test, resize,shape, shallow, attention):
    print("Load dataset")

    global pathTrainImage
    pathTrainImage = trainImage
    global pathTestImage
    pathTestImage = testingImage
    global testGlobal

    global pathTrainMask
    pathTrainMask = trainMask
    global pathTestMask
    pathTestMask = testMask

    global resizeGlobal
    resizeGlobal=resize

    global attentionGlobal
    attentionGlobal = attention
    global shapeImages
    shapeImages = shape

    global Name
    shallow_list = str(shallow).replace('[', '').replace(']', '')
    Name = name+f'{shallow_list}'
    if attentionGlobal:
        Name = Name + '_attention'

    global shallowList
    shallowList=shallow
    

 





    import image_generator_prediction
    test = image_generator_prediction.ImageMaskGenerator(
        images_folder=pathTestImage,
        masks_folder=pathTestMask,
        batch_size=size_test,
        nb_classes=2, split=0, train=False, resize=resize, size=shapeImages
    )

    testGlobal=test



    trials = Trials()

    hyperparams = {"batch": hp.choice("batch", [4, 8, 16,32, 64]),
                   "augmentation": hp.choice("augmentation", ["True", "False"]),
                   'dropout': hp.uniform("dropout", 0, 1),
                   "learning_rate": hp.uniform("learning_rate", 0.0001, 0.001),"alpha": hp.uniform('alpha', 0, 1),
                   "temperature": hp.uniformint('temperature', 1, 30, q=1)}


    fmin(hyperopt_fcn, hyperparams, trials=trials, algo=tpe.suggest, max_evals=35)

    print("done")
    return best_model


def hyperopt_fcn(params):
    global SavedParameters
    global best_model
    start_time = time.time()

    print("start train")
    global iteration


    model, val = NN(pathTrainImage, pathTrainMask, params)

    time_training = time.time() - start_time

    print("start predict")

    start_time = time.time()


    if resizeGlobal:
        YTestGlobal, Y_predicted, Y_predicted_teacher=Utils.predictionWithResizeSelf2(pathTestMask, testGlobal[0][0], model, input_shape=shapeImages)

    else:
        Y_predicted,middle_output1, middle_output2, middle_output3, final_fea, middle1_fea, middle2_fea, middle3_fea = model.predict(testGlobal[0][0], verbose=0, use_multiprocessing=True, workers=12)
        #YTestGlobal = testGlobal[0][1].ravel()
        #Y_predicted = ((Y_predicted > 0.5) + 0).ravel()



    time_predict = time.time() - start_time

    precision_macro_t, recall_macro_t, fscore_macro_t, support = precision_recall_fscore_support(YTestGlobal,
                                                                                                 Y_predicted,
                                                                                                 average='macro')
    precision_micro_t, recall_micro_t, fscore_micro_t, support = precision_recall_fscore_support(YTestGlobal,
                                                                                                 Y_predicted,
                                                                                                 average='micro')
    precision_weighted_t, recall_weighted_t, fscore_weighted_t, support = precision_recall_fscore_support(YTestGlobal,
                                                                                                          Y_predicted,
                                                                                                          average='weighted')

    accuracy_t = accuracy_score(YTestGlobal, Y_predicted)
    cf = confusion_matrix(YTestGlobal, Y_predicted)
    r = Utils.res(cf)
    tn, fp, fn, tp = cf.ravel()
    iou=tp/(tp+fn+fp)


    cf_teacher = confusion_matrix(YTestGlobal, Y_predicted_teacher)
    rTeacher = Utils.res(cf_teacher)
    tn_t, fp_t, fn_t, tp_t = cf_teacher.ravel()
    iouTeacher=tp_t/(tp_t+fn_t+fp_t)

    K.clear_session()

    SavedParameters.append(val)

    global best_val_acc
    global best_test_acc
    global best_val_loss

    SavedParameters[-1].update(
        {"precision_macro_t": precision_macro_t, "recall_macro_t": recall_macro_t, "fscore_macro_t": fscore_macro_t,
         "precision_micro_t": precision_micro_t, "recall_micro_t": recall_micro_t, "fscore_micro_t": fscore_micro_t,
         "precision_weighted_t": precision_weighted_t, "recall_weighted_t": recall_weighted_t,
         "fscore_weighted_t": fscore_weighted_t,
         "accuracy_t": accuracy_t, "IOU_test": iou,"TP_test": tp,
          "FN_test": fn, "FP_test": fp, "TN_test": tn,
         "time_training": time_training, "time_predict": time_predict, "augmentation": params["augmentation"],
         "learning_rate": params["learning_rate"], "batch": params["batch"],"alpha": params["alpha"],
         "temperature": params["temperature"], "dropout":params['dropout']})

    SavedParameters[-1].update({
        "OA_test": r[0],
        "P_test": r[2],
        "R_test": r[3],
        "F1_test": r[4],
        "FAR_test": r[5],
        "TPR_test": r[6]})

    SavedParameters[-1].update({
        "Teacher_TP_test": tp_t,
        "Teacher_FN_test": fn_t, "Teacher_FP_test": fp_t, "Teacher_TN_test_t": tn_t,
        "Teacher_OA_test": rTeacher[0],
        "Teacher_F1_test": rTeacher[4], "Teacher_IOU_test": iouTeacher, "iteration": iteration})

    
    if SavedParameters[-1]["F1_val"] > best_val_acc:
        print("new saved model:" + str(SavedParameters[-1]))
        best_model = model
        import os
        # model.save(Name.replace(".csv", "_model.h5"))
        #model.save(Name+'_model.h5')
        model.save(Name+'_model.tf')
        model.save_weights(Name+"_weights.h5")


        best_val_acc = SavedParameters[-1]["F1_val"]
      

    SavedParameters = sorted(SavedParameters, key=lambda i: float('-inf') if math.isnan(i['F1_val']) else i['F1_val'],
                             reverse=True)

    try:
        with open(Name + '_Results.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
            writer.writeheader()
            writer.writerows(SavedParameters)
    except IOError:
        print("I/O error")

    return {'loss': -val["F1_val"], 'status': STATUS_OK}
    iteration=iteration+1
    


def NN(pathTrainImage, pathTrainMask, params):
    print(params)
    model = SatelliteSelfUnet.satellite_unet(shapeImages, shallowList, attention=attentionGlobal, dropout=params['dropout'])





    ####implement load ###
    batch_size = params['batch']
    # from keras.utils.vis_utils import plot_model
    generator = image_generator.ImageMaskGenerator(
        images_folder=pathTrainImage,
        masks_folder=pathTrainMask,
        batch_size=batch_size,
        nb_classes=2, augmentation=params['augmentation'],resize=resizeGlobal, size=shapeImages
    )

    valid = image_generator.ImageMaskGenerator(
        images_folder=pathTrainImage,
        masks_folder=pathTrainMask,
        batch_size=batch_size,
        nb_classes=2,
        validation=True,resize=resizeGlobal,size=shapeImages
    )

    save_model_band_attention = [EarlyStopping(patience=20, verbose=1, monitor="val_loss")]



    metrics = [losses.dice_coef_self,losses.accuracy_teacher, losses.f1, losses.accuracy]


   # model.add_loss(losses.self_loss)
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=params['learning_rate']),
        metrics=metrics, loss=self_loss(alpha=params['alpha'], temperature=params['temperature']) #beta=params['lambda']
    )


    #model.compile(, loss=loss, metrics=metrics)
    hist=model.fit(generator, epochs=150, validation_data=valid,
              callbacks=[save_model_band_attention])

    np.save(Name + '-history.npy', model.history.history)


    x_val=valid[0][0]
    y_val=(valid[0][1]).ravel()
    print(x_val.shape)
    print(y_val.shape)

    y_pred = model.predict(x_val, verbose=0, use_multiprocessing=True, workers=12)



    sum = 0
    for j in range(y_pred.shape[3]):
        sum = sum + y_pred[:, :, :, j]
    Y_predicted_val=y_pred[:, :, :, 0]
    Y_predicted_valSum = sum / y_pred.shape[3]
    Y_predicted_sum = (( Y_predicted_valSum > 0.5) + 0).ravel()
    Y_predicted = ((Y_predicted_val > 0.5) + 0).ravel()



    x_train=generator[0][0]
    y_train=generator[0][1].ravel()
    y_pred = model.predict(x_train, verbose=0, use_multiprocessing=True, workers=12)
    sum = 0
    for j in range(y_pred.shape[3]):
        sum = sum + y_pred[:, :, :, j]
    Y_predicted_train = sum / y_pred.shape[3]
    Y_predicted_train = ((Y_predicted_train > 0.5) + 0).ravel()
    cf = confusion_matrix(y_train, Y_predicted_train)
    r = Utils.res(cf)
    if (len(cf) > 1):
        tn, fp, fn, tp = cf.ravel()
    else:
        tn = cf[0][0]
        tp = 0
        fp = 0
        fn = 0

    cfVal = confusion_matrix(y_val, Y_predicted_sum)
    rVal_all = Utils.res(cfVal)

    #qui
    cfVal = confusion_matrix(y_val, Y_predicted)
    rVal = Utils.res(cfVal)



    precision_macro_train, recall_macro_train, fscore_macro_train, support = precision_recall_fscore_support(y_train,
                                                                                                       Y_predicted_train,
                                                                                                       average='macro')
    precision_micro_train, recall_micro_train, fscore_micro_train, support = precision_recall_fscore_support(y_val,
                                                                                                       Y_predicted,
                                                                                                       average='micro')
    precision_weighted_train, recall_weighted_train, fscore_weighted_train, support = precision_recall_fscore_support(y_val,
                                                                                                                Y_predicted,
                                                                                                                average='weighted')
    accuracy_train = accuracy_score(y_train, Y_predicted_train)



    precision_macro_val, recall_macro_val, fscore_macro_val, support = precision_recall_fscore_support(y_val,
                                                                                                       Y_predicted,
                                                                                                       average='macro')
    precision_micro_val, recall_micro_val, fscore_micro_val, support = precision_recall_fscore_support(y_val,
                                                                                                       Y_predicted,
                                                                                                       average='micro')
    precision_weighted_val, recall_weighted_val, fscore_weighted_val, support = precision_recall_fscore_support(y_val,
                                                                                                                Y_predicted,
                                                                                                                average='weighted')
    accuracy_val = accuracy_score(y_val, Y_predicted)

    del support
    epoches = len(hist.history['val_loss'])
    print(hist.history['val_loss'])
    min_val_loss = np.amin(hist.history['val_loss'])
    print(min_val_loss)


    return model, {"val_loss": min_val_loss, "F1_val":rVal_all[4], "F1_val_teacher":rVal[4], "P_val": rVal_all[2],"R_val": rVal_all[3],"TP_train": tp,
          "FN_train": fn, "FP_train": fp, "TN_train": tn, "OA_train": r[0],
         "P_train": r[2],"R_train": r[3],"F1_train": r[4],"precision_macro_train": precision_macro_train,
                   "recall_macro_train": recall_macro_train, "fscore_macro_train": fscore_macro_train,
                   "precision_micro_train": precision_micro_train, "recall_micro_train": recall_micro_train,
                   "fscore_micro_train": fscore_micro_train,
                   "precision_weighted_train": precision_weighted_train, "recall_weighted_train": recall_weighted_train,
                   "fscore_weighted_train": fscore_weighted_train,
                   "accuracy_train": accuracy_train, "precision_macro_val": precision_macro_val,
                   "recall_macro_val": recall_macro_val, "fscore_macro_val": fscore_macro_val,
                   "precision_micro_val": precision_micro_val, "recall_micro_val": recall_micro_val,
                   "fscore_micro_val": fscore_micro_val,
                   "precision_weighted_val": precision_weighted_val, "recall_weighted_val": recall_weighted_val,
                   "fscore_weighted_val": fscore_weighted_val,
                   "accuracy_val": accuracy_val, "epochs": epoches}
