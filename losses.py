import tensorflow.keras.backend as K
import tensorflow as tf


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) +
                                           smooth)


def dice_coef_self(y_true, y_pred, smooth=1):
    y_pred = y_pred[:, :, :, 0]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) +
                                           smooth)


def accuracy_teacher(y_true, y_pred):
    y_pred = y_pred[:, :, :, 0]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = tf.cast((y_pred_f > 0.5), tf.float32)
    true_pos = K.sum(y_true_f * y_pred_f)
    true_neg= K.sum((1-y_true_f) * (1-y_pred_f))
    #tf.print("true pos :", true_pos)
    #tf.print("true neg:", true_neg)
    false_neg = K.sum(y_true_f * (1 - y_pred_f))
    false_pos = K.sum((1 - y_true_f) * y_pred_f)
    return (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)

def accuracy(y_true, y_pred):
    y_pred_list =0
    for j in range(y_pred.shape[3]):
        y_pred_list +=K.flatten(y_pred[:, :, :, j])

    y_pred_f=y_pred_list/y_pred.shape[3]

    y_pred_f = tf.cast((y_pred_f > 0.5), tf.float32)
    y_true_f = K.flatten(y_true)
    #y_pred_f = K.flatten(y_pred)
    true_pos = K.sum(y_true_f * y_pred_f)
    true_neg= K.sum((1-y_true_f) * (1-y_pred_f))
    #tf.print("true pos :", true_pos)
    #tf.print("true neg:", true_neg)
    false_neg = K.sum(y_true_f * (1 - y_pred_f))
    false_pos = K.sum((1 - y_true_f) * y_pred_f)
    return (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)

def calculate_ensemble(p1,p2,p3,p4):
    sum=p1+p2+p3+p4
    #tf.print("sum",sum/4)

    return sum/4

def f1(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_list = 0
    for j in range(y_pred.shape[3]):
        y_pred_list +=K.flatten(y_pred[:, :, :, j])
    y_pred_f=y_pred_list/y_pred.shape[3]
    '''
    y_pred1 = y_pred[:, :, :, 0]
    middle_output1 = y_pred[:, :, :, 1]
    middle_output2 = y_pred[:, :, :, 2]
    middle_output3 = y_pred[:, :, :, 3]
    y_true_f = K.flatten(y_true)
    y_pred_f1 = K.flatten(y_pred1)
    y_pred_f2 = K.flatten(middle_output1)
    y_pred_f3 = K.flatten(middle_output2)
    y_pred_f4 = K.flatten(middle_output3)


    y_pred_f=calculate_ensemble(y_pred_f1,y_pred_f2, y_pred_f3,y_pred_f4)
    '''

    y_pred_f=tf.cast((y_pred_f > 0.5),tf.float32)


    p = prec(y_true_f, y_pred_f)
    r = rec(y_true_f, y_pred_f)
    F1=2 * ((p * r) / (p + r + K.epsilon()))

    return F1

def prec(y_true, y_pred):
    true_pos = K.sum(y_true * y_pred)
    false_pos = K.sum((1 - y_true) * y_pred)
    return true_pos / ((true_pos+ false_pos) + K.epsilon())
    
def rec(y_true, y_pred):
    true_pos = K.sum(y_true * y_pred)
    false_neg = K.sum(y_true * (1 - y_pred))
    return true_pos / ((true_pos+ false_neg) + K.epsilon())

def recall(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)

    # tf.print("true pos :", true_pos)
    # tf.print("false neg : ", false_neg)
    # tf.print("false pos : ", false_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg +
                                  (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)



'''
def self_loss(alpha=0.1, temperature=10, beta=2):
    def loss(y_true,y_pred):
        y_true=K.flatten(y_true)

        output = y_pred[:, :, :, 0]
        middle_output1=y_pred[:,:,:,1]
        middle_output2=y_pred[:,:,:,2]
        middle_output3=y_pred[:,:,:,3]
        
        
        
        final_fea=y_pred[:, :,:,4:68 ]
        middle1_fea=y_pred[:, :,:,68:132 ]
        middle2_fea=y_pred[:, :,:,132:196 ]
        middle3_fea=y_pred[:, :,:,196:260 ]

        kl = tf.keras.losses.KLDivergence()
        ce = tf.keras.losses.BinaryCrossentropy()


        output=K.flatten(output)
        #tf.print("output", output)
        #tf.print("output2", middle_output1)
        #tf.print("output3",middle_output2)
        #print("----")
        #tf.print("true", y_true)

        middle_output1 = K.flatten(middle_output1)
        middle_output2 = K.flatten(middle_output2)
        middle_output3 = K.flatten(middle_output3)


        teacher_loss = tversky_loss(y_true, output)
        middle1_loss = tversky_loss(y_true, middle_output1)
        middle2_loss = tversky_loss(y_true, middle_output2)
        middle3_loss = tversky_loss(y_true, middle_output3)



        output = tf.nn.softmax(output / temperature)
        middle_output1= tf.nn.softmax(middle_output1/temperature)
        middle_output2 =tf.nn.softmax(middle_output2/temperature)
        middle_output3 = tf.nn.softmax(middle_output3/temperature)
        loss1_kd = kl(output, middle_output1)
        loss2_kd = kl(output, middle_output2)
        loss3_kd = kl(output, middle_output3)
        sum_kl =(loss1_kd+loss2_kd+loss3_kd)/3
        #tf.print("sum", sum_kl)

        middle1_fea=K.flatten(middle1_fea)
        middle2_fea = K.flatten(middle2_fea)
        middle3_fea = K.flatten(middle3_fea)
        final_fea = K.flatten(final_fea)
        feature_loss1 = feature_loss_function(middle1_fea, final_fea)

        feature_loss2 = feature_loss_function(middle1_fea, final_fea)
        feature_loss3 = feature_loss_function(middle2_fea, final_fea)
        tf.print(teacher_loss)

        sum_ce=(teacher_loss+middle1_loss+ middle2_loss+ middle3_loss)/4

        penalty_loss = beta * ((feature_loss1 + feature_loss2 + feature_loss3)/3)




        loss_kl = alpha * sum_kl
        loss_ce = (1 - alpha) * sum_ce
        loss=loss_ce+loss_kl
        #loss=loss_ce+loss_kl+penalty_loss
        #tf.print(loss)
        return loss
    return loss
    '''

def self_loss(alpha, temperature):
    def loss(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred_list=[]
        for j in range(y_pred.shape[3]):
            y_pred_list.append(y_pred[:,:,:, j])


        output = y_pred_list[0]



        kl = tf.keras.losses.KLDivergence()
        ce = tf.keras.losses.BinaryCrossentropy()

        output = K.flatten(output)
        # tf.print("output", output)
        # tf.print("output2", middle_output1)
        # tf.print("output3",middle_output2)
        # print("----")
        # tf.print("true", y_true)
        teacher_loss = tversky_loss(y_true, output)
        sum_ce=teacher_loss
        sum_kl=0
        output = tf.nn.softmax(output / temperature)
        for p in range(1,len(y_pred_list)):
            middle_prediction=K.flatten(y_pred[:,:,:, p])
            middle_loss = tversky_loss(y_true, middle_prediction)
            sum_ce+=middle_loss
            middle_output = tf.nn.softmax(middle_prediction / temperature)
            loss_kd = kl(output, middle_output)
            sum_kl+=loss_kd

        sum_ce = sum_ce / y_pred.shape[3]
        sum_kl=sum_kl / (y_pred.shape[3]-1)

        loss_kl = alpha * sum_kl
        loss_ce = (1 - alpha) * sum_ce
        loss = loss_ce + loss_kl
        # loss=loss_ce+loss_kl+penalty_loss
        # tf.print(loss)
        return loss

    return loss
def feature_loss_function(fea, target_fea):
    #intermediate=tf.cast(((fea > 0) | (target_fea > 0)), tf.float32)
    #tf.print(intermediate)
    return tf.norm(fea-target_fea, ord='euclidean')
    #loss = (tf.math.abs(fea - target_fea)**2)
    #return tf.math.sqrt(loss)

