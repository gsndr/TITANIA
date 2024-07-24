# https://cdn-sv1.deepsense.ai/wp-content/uploads/2017/04/architecture_details.png
# https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/

from channel_attention import EfficientChannelAttention2D as Attention


TF=1

if TF:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        BatchNormalization, Conv2D, Conv2DTranspose,
        MaxPooling2D, UpSampling2D, Input,
        concatenate, Activation, Dropout, Concatenate
    )
else:
    from keras.models import Model
    from keras.layers import (
        BatchNormalization, Conv2D, Conv2DTranspose,
        MaxPooling2D, UpSampling2D, Input,
        concatenate
    )


def bn_conv_relu(input, filters, bachnorm_momentum, **conv2d_args):
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2D(filters, **conv2d_args)(x)
    return x


def bn_upconv_relu(input, filters, bachnorm_momentum, **conv2d_trans_args):
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2DTranspose(filters, **conv2d_trans_args)(x)
    return x

def shallow(input, name, num_filters, num_transpose, dropout=0.5, l2=0, attention=False):
    if num_transpose>0:
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same", name=name+'conv2dTranspose')(input)
        for i in range(num_transpose-1):
            x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same", name=name + 'conv2dTranspose'+str(i))(x)
    else:
        x = input

def satellite_unet(
        input_shape, shallowList,
        num_classes=1,
        output_activation='sigmoid',
        num_layers=4, attention=0, dropout=0.5):
    inputs = Input(input_shape)

    filters = 64
    upconv_filters = 96

    kernel_size = (3, 3)
    activation = 'relu'
    strides = (1, 1)
    padding = 'same'
    kernel_initializer = 'he_normal'

    conv2d_args = {
        'kernel_size': kernel_size,
        'activation': activation,
        'strides': strides,
        'padding': padding,
        'kernel_initializer': kernel_initializer
    }

    conv2d_trans_args = {
        'kernel_size': kernel_size,
        'activation': activation,
        'strides': (2, 2),
        'padding': padding,
        'output_padding': (1, 1)
    }

    bachnorm_momentum = 0.01

    pool_size = (2, 2)
    pool_strides = (2, 2)
    pool_padding = 'valid'

    maxpool2d_args = {
        'pool_size': pool_size,
        'strides': pool_strides,
        'padding': pool_padding,
    }





    x = Conv2D(filters, **conv2d_args, name='conv0')(inputs)
    c1 = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(c1, filters, bachnorm_momentum, **conv2d_args)
    x = MaxPooling2D(**maxpool2d_args)(x)

    down_layers = []

    for l in range(num_layers):
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        down_layers.append(x)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = MaxPooling2D(**maxpool2d_args)(x)

    # upsampling 1
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    d0 = bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)

   #upsampling 2
    if attention:
        x = Attention(inputs.shape[-1])(down_layers[3])
        x = concatenate([d0, x])
    else:
        x = concatenate([d0, down_layers[3]])
    x = bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    d1 = bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)

    # upsampling 3
    if attention:
        x = Attention(inputs.shape[-1])(down_layers[2])
        x = concatenate([d1, x])
    else:
        x = concatenate([d1, down_layers[2]])
    x = bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    d2 = bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)

    # upsampling 4
    if attention:
        x = Attention(inputs.shape[-1])(down_layers[1])
        x = concatenate([d2, x])
    else:
        x = concatenate([d2, down_layers[1]])
    x = bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    d3 = bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)

    # upsampling 5
    if attention:
        x = Attention(inputs.shape[-1])(down_layers[0])
        x = concatenate([d3, x])
    else:
        x = concatenate([d3, down_layers[0]])
    x = bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    d4 = bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)


    if attention:
        x = Attention(inputs.shape[-1])(c1)
        x = concatenate([d4, x])
    else:
        x = concatenate([d4, c1])
    x = bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)


    shallow_layers=[]

    # shallow layer
    if 5 in shallowList:
        s4_middle = Conv2D(64, (1, 1), strides=1, padding="same", name='s4_conv2d0')(d4)
        s4 = BatchNormalization(momentum=bachnorm_momentum)(s4_middle)
        s4 = Activation("relu")(s4)
        s4 = Dropout(dropout)(s4)
        output_4=Conv2D(num_classes, kernel_size=(1, 1), strides=(1, 1), activation=output_activation, padding='valid')(s4)
        shallow_layers.append(output_4)

    # shallow layer 3
    if 4 in shallowList:
        input_s3 = Conv2DTranspose(64, (2, 2), strides=2, padding="same")(d3)
        s3_middle = Conv2D(64, (1, 1), strides=1, padding="same", name='s3_conv2d0')(input_s3)
        s3 = BatchNormalization(momentum=bachnorm_momentum)(s3_middle)
        s3 = Activation("relu")(s3)
        s3 = Dropout(dropout)(s3)
        output_3 = Conv2D(num_classes, kernel_size=(1, 1), strides=(1, 1), activation=output_activation, padding='valid')(
            s3)
        shallow_layers.append(output_3)



    # shallow layer 2
    if 3 in shallowList:
        input_s2 = Conv2DTranspose(64, (2, 2), strides=2, padding="same")(d2)
        input_s2 = Conv2DTranspose(64, (2, 2), strides=2, padding="same")(input_s2)
        s2_middle = Conv2D(64, (1, 1), strides=1, padding="same", name='s2_conv2d0')(input_s2)
        s2 = BatchNormalization(momentum=bachnorm_momentum)(s2_middle)
        s2 = Activation("relu")(s2)
        s2 = Dropout(dropout)(s2)
        output_2 = Conv2D(num_classes, kernel_size=(1, 1), strides=(1, 1), activation=output_activation, padding='valid')(
            s2)
        shallow_layers.append(output_2)

    if 2 in shallowList:
        input_s1 = Conv2DTranspose(64, (2, 2), strides=2, padding="same")(d1)
        input_s1 = Conv2DTranspose(64, (2, 2), strides=2, padding="same")(input_s1)
        input_s1 = Conv2DTranspose(64, (2, 2), strides=2, padding="same")(input_s1)
        s1_middle = Conv2D(64, (1, 1), strides=1, padding="same", name='s1_conv2d0')(input_s1)
        s1 = BatchNormalization(momentum=bachnorm_momentum)(s1_middle)
        s1 = Activation("relu")(s1)
        s1 = Dropout(dropout)(s1)
        output_1 = Conv2D(num_classes, kernel_size=(1, 1), strides=(1, 1), activation=output_activation, padding='valid')(
            s1)
        shallow_layers.append(output_1)


    if 1 in shallowList:
        input_s0 = Conv2DTranspose(64, (2, 2), strides=2, padding="same")(d0)
        input_s0 = Conv2DTranspose(64, (2, 2), strides=2, padding="same")(input_s0)
        input_s0 = Conv2DTranspose(64, (2, 2), strides=2, padding="same")(input_s0)
        input_s0 = Conv2DTranspose(64, (2, 2), strides=2, padding="same")(input_s0)
        s0_middle = Conv2D(64, (1, 1), strides=1, padding="same", name='s0_conv2d0')(input_s0)
        s0 = BatchNormalization(momentum=bachnorm_momentum)(s0_middle)
        s0 = Activation("relu")(s0)
        s0 = Dropout(dropout)(s0)
        output_0 = Conv2D(num_classes, kernel_size=(1, 1), strides=(1, 1), activation=output_activation, padding='valid')(
            s0)
        shallow_layers.append(output_0)




    outputs = Conv2D(num_classes, kernel_size=(1, 1), strides=(1, 1), activation=output_activation, padding='valid')(x)
    shallow_layers.append(outputs)
    outLayer = Concatenate(axis=-1)(shallow_layers)


    model = Model(inputs=[inputs], outputs=outLayer)
    return model


