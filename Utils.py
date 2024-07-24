import numpy as np
from Preprocessing import Preprocessing
def res(cm):
    '''
    tp = cm[1][1]  # attacks true
    fn = cm[1][0]  # attacs predict normal
    fp = cm[0][1]  # normal predict attacks
    tn = cm[0][0]  # normal as normal
    '''
    if (len(cm) > 1):
        tn, fp, fn, tp=cm.ravel()

    else:
        tn=cm[0][0]
        tp=0
        fp=0
        fn=0

    attacks = tp + fn
    normals = fp + tn
    OA = (tp + tn) / (attacks + normals)
    if (len(cm) > 1):
        AA = ((tp / attacks) + (tn / normals)) / 2
        P = tp / (tp + fp)
        R = tp / (tp + fn)
        F1 = 2 * ((P * R) / (P + R))
        TPR = tp / (tp + fn)
    else:
        AA = (0 + (tn / normals)) / 2
        P=0
        R=0
        F1=0
        TPR=0
    FAR = fp / (fp + tn)

    r = [OA, AA, P, R, F1, FAR, TPR]
    return r

def predictionWithResizeSelf2(path,images, model, input_shape=(32,32,12)):
    allTrue=[]
    allPred=[]
    allPred_teacher=[]
    import os
    masks = []
    for root, _, files in os.walk(path):
        files.sort()
        for file in files:
            m = np.load(os.path.join(root, file))
            masks.append(m)
    print(len(masks))
    for i in range(len(images)):
        y_pred = model.predict(images[i].reshape(1, input_shape[0], input_shape[1], input_shape[2]))
        y_pred_teacher=y_pred[:,:,:, 0]
        sum=0
        for j in range(y_pred.shape[3]):
            sum=sum+y_pred[:,:,:, j]
        pred=sum/y_pred.shape[3]


        true=masks[i]
        prep=Preprocessing()
        pred = prep.reduce_padding2(pred, true)
        pred_teacher = prep.reduce_padding2(y_pred_teacher, true)
        pred_teacher = ((pred_teacher > 0.5) + 0).ravel()



        print(pred.shape)

        pred = ((pred > 0.5) + 0).ravel()
        true = true.flatten()
        allTrue.append(true)
        allPred.append(pred)
        allPred_teacher.append(pred_teacher)
    allPred = [arr.tolist() for arr in allPred]
    allPred=flattenList(allPred)
    allPred_teacher = [arr.tolist() for arr in allPred_teacher]
    allPred_teacher = flattenList(allPred_teacher)
    allTrue = [arr.tolist() for arr in allTrue]
    allTrue = flattenList(allTrue)
    return allTrue,allPred, allPred_teacher




def predictionWithResizeSelfNoList(mask,images, model, input_shape=(32,32,12)):
    prep = Preprocessing()
    y_pred = model.predict(images.reshape(1, input_shape[0], input_shape[1], input_shape[2]))
    y_pred_teacher=y_pred[:,:,:, 0]
    sum=0
    sum_s = 0
    pred_Students=[]
    for j in range(y_pred.shape[3]):
        pred_s = y_pred[:, :, :, j]
        true = mask
        pred_s = prep.reduce_padding2(pred_s, true)
        pred_s = ((pred_s > 0.5) + 0).ravel()
        pred_Students.append(pred_s)
        sum = sum + y_pred[:, :, :, j]
        if j > 0:
            sum_s = sum_s + y_pred[:, :, :, j]
    pred = sum / y_pred.shape[3]
    pred_s = sum_s / (y_pred.shape[3] - 1)


    true=mask

    #total prediction
    pred = prep.reduce_padding2(pred, true)
    pred = ((pred > 0.5) + 0).ravel()

    #pred teacher
    pred_teacher = prep.reduce_padding2(y_pred_teacher, true)
    pred_teacher = ((pred_teacher > 0.5) + 0).ravel()


    print(pred.shape)





    return pred, pred_teacher
def flattenList(l):
    return [item for sublist in l for item in sublist]




def print_results(path, y_true, y_pred,  model, non_rounded_y_pred=None, write=True):
    """
      FUnction saving the classification report of the single outputs classifiers in  .txt format
      :param path: path to save the report
      :param y_true: true y
      :param y_pred: vpredicted y
      :param write: booleano, if true write the file, otherwise return only the classification report
      :return: classification report if write equal to false otherwise void
      """
    from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score, roc_auc_score, roc_curve

    cm = confusion_matrix(y_true, y_pred)


    val = ''
    val = val + ('\n****** ******\n\n')
    val = val + (classification_report(y_true, y_pred))
    val = val + '\n\n----------- f1 macro ---------------\n'
    val = val + str(f1_score(y_true, y_pred, average='macro'))
    val = val + '\n\n----------- f1 micro ---------------\n'
    val = val + str(f1_score(y_true, y_pred, average='micro'))
    val = val + '\n\n----------- f1 weighted ---------------\n'
    val = val + str(f1_score(y_true, y_pred, average='weighted'))
    val = val + '\n\n----------- OA ---------------\n'
    val = val + str(accuracy_score(y_true, y_pred,))
    val = val + '\n\n----------- Confusion matrix ---------------\n'
    val = val + str(cm)
    val = val + '\n\n----------- tn, fp, fn, tp ---------------\n'
    val = val + str(cm.ravel())
    val = val + '\n\n----------- IOU ---------------\n'
    tn, fp, fn, tp = cm.ravel()
    val = val + str(tp / (tp + fn + fp))

    r=res(cm)
    val = val + '\n\n----------- OA  manually ---------------\n'
    val = val + str(r[0])
    val = val + '\n\n----------- AA  manually ---------------\n'
    val = val + str(r[1])
    val = val + '\n\n----------- P  attack ---------------\n'
    val = val + str(r[2])
    val = val + '\n\n----------- R  attack ---------------\n'
    val = val + str(r[3])
    val = val + '\n\n----------- F1  attack ---------------\n'
    val = val + str(r[4])
    val = val + '\n\n----------- FAR ---------------\n'
    val = val + str(r[5])
    val = val + '\n\n----------- TPR ---------------\n'
    val = val + str(r[6])



    if non_rounded_y_pred is not None:
        fpr, tpr, thresholds = roc_curve(y_true, non_rounded_y_pred)
        auc_value = roc_auc_score(y_true, non_rounded_y_pred)
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(5, 5), dpi=100)
        plt.plot(fpr, tpr, linestyle='-', marker='.', label="(auc = %0.4f)" % auc_value)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        fig.savefig(path+'auc-roc.png', bbox_inches='tight')
        val = val + '\n\n----------- AUC-ROC ---------------\n'
        val = val + str(auc_value)





    if write:
        with open(path + '_results.txt', 'w', encoding='utf-16') as file:
            file.write(val)
            file.write('\n\n----------- Summary ---------------\n')
            model.summary(print_fn=lambda x: file.write(x + '\n'))


    else:
        return val



