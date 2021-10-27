"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021).
EfficientNetV2: Smaller Models and Faster Training
arXiv preprint arXiv:2104.00298.
"""
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    PReLU,
    Reshape,
    Multiply,
)
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 0.001
CONV_KERNEL_INITIALIZER = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'glorot_uniform'

BLOCK_CONFIGS = {
    "b0": {  # width 1.0, depth 1.0
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 32, 48, 96, 112, 192],
        "depthes": [1, 2, 2, 3, 5, 8],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "b1": {  # width 1.0, depth 1.1
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 32, 48, 96, 112, 192],
        "depthes": [2, 3, 3, 4, 6, 9],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "b2": {  # width 1.1, depth 1.2
        "first_conv_filter": 32,
        "output_conv_filter": 1408,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 32, 56, 104, 120, 208],
        "depthes": [2, 3, 3, 4, 6, 10],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "b3": {  # width 1.2, depth 1.4
        "first_conv_filter": 40,
        "output_conv_filter": 1536,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 40, 56, 112, 136, 232],
        "depthes": [2, 3, 3, 5, 7, 12],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "s": {  # width 1.4, depth 1.8
        "first_conv_filter": 24,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [24, 48, 64, 128, 160, 256],
        "depthes": [2, 4, 4, 6, 9, 15],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "m": {  # width 1.6, depth 2.2
        "first_conv_filter": 24,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6, 6],
        "out_channels": [24, 48, 80, 160, 176, 304, 512],
        "depthes": [3, 5, 5, 7, 14, 18, 5],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "use_ses": [0, 0, 0, 1, 1, 1, 1],
    },
    "l": {  # width 2.0, depth 3.1
        "first_conv_filter": 32,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6, 6],
        "out_channels": [32, 64, 96, 192, 224, 384, 640],
        "depthes": [4, 7, 7, 10, 19, 25, 7],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "use_ses": [0, 0, 0, 1, 1, 1, 1],
    },
    "xl": {
        "first_conv_filter": 32,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6, 6],
        "out_channels": [32, 64, 96, 192, 256, 512, 640],
        "depthes": [4, 8, 8, 16, 24, 32, 8],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "use_ses": [0, 0, 0, 1, 1, 1, 1],
    },
}

#加横线表示内部函数，希望能够在类里被调用
def _make_divisible(v, divisor=4, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", name=""):
    '''
    这个函数将会对inputs进行Conv2D，不使用偏置，且卷积核初始化为CONV_KERNEL_INITIALIZER
    '''
    return Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "conv")(
        inputs
    )


def batchnorm_with_activation(inputs, activation="swish", name=""):
    """Performs a batch normalization followed by an activation. """
    bn_axis = 1 if K.image_data_format() == "channels_first" else -1
    nn = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=name + "bn",
    )(inputs)
    if activation:
        nn = Activation(activation=activation, name=name + activation)(nn)
        # nn = PReLU(shared_axes=[1, 2], alpha_initializer=tf.initializers.Constant(0.25), name=name + "PReLU")(nn)
    return nn


def se_module_by_conv2d(inputs, se_ratio=4, name=""):
    filters = inputs.shape[-1]#图片的通道数
    reduction = filters // se_ratio
    # 计算axis上的元素平均值，全局平均池化
    se = tf.reduce_mean(inputs,(1,2),keepdims=True)
    #这儿的两个Conv2D其实就是一种全连接层。
    se = Conv2D(reduction, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "1_conv")(se)
    se = Activation("swish")(se)
    se = Conv2D(filters, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "2_conv")(se)
    se = Activation("sigmoid")(se)
    return Multiply()([inputs, se])

def se_module_by_dense(inputs,se_ratio=4,name=""):
    filters = inputs.shape[-1]#图片的通道数
    reduction = filters // se_ratio
    #计算axis上的元素平均值，全局均值池化
    se = tf.math.reduce_mean(inputs,(1,2),keepdims=True)
    se = tf.keras.layers.Dense(reduction,activation="swish",use_bias=True)(se)
    se = tf.keras.layers.Dense(filters,activation='sigmoid',use_bias=True)(se)
    return tf.keras.layers.Multiply()([inputs,se])

class SE(tf.keras.layers):
    def __init__(self,inputs,se_ratio:int = 4,channel_pos=-1,name:str=""):
        '''
        这个函数是使用Conv1x1实现的SE模块，并使用reduc_mean实现GlobalAveragePooling
        channel_pos：通道的维数位于第二维度还是最后一维度
        '''
        super(SE,self).__init__()
        self.inputs = inputs
        self.se_ratio = se_ratio
        self.name = name
        self.ch_pos = channel_pos
        self.filters = self.inputs.shape[-1]
        self.reduction = self.filters // self.se_ratio
        self.conv1 = Conv2D(self.reduction,1,1,use_bias=True,kernel_initializer=CONV_KERNEL_INITIALIZER,name=self.name+'1_conv')
        self.act1 = Activation('swish')
        self.conv2 = Conv2D(self.filters,1,1,use_bias=True,kernel_initializer=CONV_KERNEL_INITIALIZER,name=self.name+'2_conv')
        self.act2 = Activation('sigmoid')
        self.multiply = Multiply()
    def call(self):
        x = tf.reduce_mean(self.inputs,(1,2),keepdims=True)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        out = self.multiply([x,self.inputs])
        return out

def MBConv(inputs, output_channel, stride, expand_ratio, shortcut, survival=None, use_se=0, is_fused=True, name=""):
    '''
    根据论文完全编写的MBConv结构
    '''
    input_channel = inputs.shape[-1] #输入通道在最后一维

    if is_fused:#如果使用Fused-MBConv
        nn = Conv2D(input_channel*expand_ratio,(3,3),strides=stride,use_bias=False,kernel_initializer=CONV_KERNEL_INITIALIZER,padding='same',name="Fused Conv3x3")(inputs)
        nn = BatchNormalization(momentum=BATCH_NORM_DECAY,epsilon=BATCH_NORM_EPSILON,name="Fused Conv3x3 BN")(nn)
        nn = Activation(activation='swish',name="Fused Conv3x3 Activate")(nn)
    elif not is_fused:#如果使用MBConv
        nn = Conv2D(input_channel*expand_ratio,(1,1),padding='same',use_bias=False,kernel_initializer=CONV_KERNEL_INITIALIZER,name="1x1 expand Convolution")(inputs)
        nn = BatchNormalization(momentum=BATCH_NORM_DECAY,epsilon=BATCH_NORM_EPSILON,name="1x1 expand Conv BN")(nn)
        nn = Activation(activation='swish',name="1x1 expand Conv activation = swish ")(nn)

        nn = DepthwiseConv2D((3,3),padding='same',strides=stride,use_bias=False,depthwise_initializer=CONV_KERNEL_INITIALIZER,name='Depwise Conv')(nn)
        nn = BatchNormalization(momentum=BATCH_NORM_DECAY,epsilon=BATCH_NORM_EPSILON,name='Depthwise Conv BN')(nn)
        nn = Activation(activation='swish',name='Depthwise Conv activ')(nn)

    if use_se:#是否使用SE模块
        nn = se_module_by_conv2d(nn, se_ratio=4 * expand_ratio, name=name + "se_")

    nn = Conv2D(output_channel,(1,1),strides=stride,padding='same',use_bias=False,kernel_initializer=CONV_KERNEL_INITIALIZER,name="降维卷积1x1")(nn)
    nn = BatchNormalization(momentum=BATCH_NORM_DECAY,epsilon=BATCH_NORM_EPSILON,name="降维卷积1x1后的BN")(nn)



    # pw-linear
    if is_fused and expand_ratio == 1:
        nn = conv2d_no_bias(nn, output_channel, (3, 3), strides=stride, padding="same", name=name + "fu_")
        nn = batchnorm_with_activation(nn, name=name + "fu_")
    else:
        nn = conv2d_no_bias(nn, output_channel, (1, 1), strides=(1, 1), padding="same", name=name + "MB_pw_")
        nn = batchnorm_with_activation(nn, activation=None, name=name + "MB_pw_")

    if shortcut:#是否使用直连
        if survival is not None and survival < 1:
            from tensorflow_addons.layers import StochasticDepth

            return StochasticDepth(float(survival))([inputs, nn])
        else:
            return Add()([inputs, nn])
    else:
        return nn


def EfficientNetV2(
    model_type,#s,m,l
    input_shape=(None, None, 3),
    num_classes=1000,
    dropout=0.2,
    first_strides=2,
    survivals=None,
    classifier_activation="softmax",
    pretrained="imagenet21k-ft1k",
    model_name="EfficientNetV2",
    kwargs=None,    # Not used, just recieving parameter
):
    blocks_config = BLOCK_CONFIGS.get(model_type.lower(), BLOCK_CONFIGS["s"])
    expands = blocks_config["expands"]
    out_channels = blocks_config["out_channels"]
    depthes = blocks_config["depthes"]
    strides = blocks_config["strides"]
    use_ses = blocks_config["use_ses"]
    first_conv_filter = blocks_config.get("first_conv_filter", out_channels[0])
    output_conv_filter = blocks_config.get("output_conv_filter", 1280)

    inputs = Input(shape=input_shape)
    out_channel = _make_divisible(first_conv_filter, 8)
    nn = conv2d_no_bias(inputs, out_channel, (3, 3), strides=first_strides, padding="same", name="stem_")
    nn = batchnorm_with_activation(nn, name="stem_")

    # StochasticDepth survival_probability values
    total_layers = sum(depthes)
    if isinstance(survivals, float):
        survivals = [survivals] * total_layers
    elif isinstance(survivals, (list, tuple)) and len(survivals) == 2:
        start, end = survivals
        survivals = [start - (1 - end) * float(ii) / total_layers for ii in range(total_layers)]
    else:
        survivals = [None] * total_layers
    survivals = [survivals[int(sum(depthes[:id])) : sum(depthes[: id + 1])] for id in range(len(depthes))]

    pre_out = out_channel
    for id, (expand, out_channel, depth, survival, stride, se) in enumerate(zip(expands, out_channels, depthes, survivals, strides, use_ses)):
        out = _make_divisible(out_channel, 8)
        is_fused = True if se == 0 else False
        for block_id in range(depth):
            stride = stride if block_id == 0 else 1
            shortcut = True if out == pre_out and stride == 1 else False
            name = "stack_{}_block{}_".format(id, block_id)
            nn = MBConv(nn, out, stride, expand, shortcut, survival[block_id], se, is_fused, name=name)
            pre_out = out

    output_conv_filter = _make_divisible(output_conv_filter, 8)
    nn = conv2d_no_bias(nn, output_conv_filter, (1, 1), strides=(1, 1), padding="valid", name="post_")
    nn = batchnorm_with_activation(nn, name="post_")

    if num_classes > 0:
        nn = GlobalAveragePooling2D(name="avg_pool")(nn)
        if dropout > 0 and dropout < 1:
            nn = Dropout(dropout)(nn)
        nn = Dense(num_classes, activation=classifier_activation, name="predictions")(nn)

    model = Model(inputs=inputs, outputs=nn, name=name)
    reload_model_weights(model, model_type, pretrained)
    return model


def reload_model_weights(model, model_type, pretrained="imagenet"):
    pretrained_dd = {"imagenet": "imagenet", "imagenet21k": "21k", "imagenet21k-ft1k": "21k-ft1k"}
    if not pretrained in pretrained_dd:
        print(">>>> No pretraind available, model will be random initialized")
        return

    pre_url = "https://github.com/leondgarse/keras_efficientnet_v2/releases/download/v1.0.0/efficientnetv2-{}-{}.h5"
    url = pre_url.format(model_type, pretrained_dd[pretrained])
    file_name = os.path.basename(url)
    try:
        pretrained_model = keras.utils.get_file(file_name, url, cache_subdir="models/efficientnetv2")
    except:
        print("[Error] will not load weights, url not found or download failed:", url)
        return
    else:
        print(">>>> Load pretraind from:", pretrained_model)
        model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)


def EfficientNetV2B0(input_shape=(224, 224, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    return EfficientNetV2(model_type="b0", model_name="EfficientNetV2B0", **locals(), **kwargs)


def EfficientNetV2B1(input_shape=(240, 240, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    return EfficientNetV2(model_type="b1", model_name="EfficientNetV2B1", **locals(), **kwargs)


def EfficientNetV2B2(input_shape=(260, 260, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    return EfficientNetV2(model_type="b2", model_name="EfficientNetV2B2", **locals(), **kwargs)


def EfficientNetV2B3(input_shape=(300, 300, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    return EfficientNetV2(model_type="b3", model_name="EfficientNetV2B3", **locals(), **kwargs)


def EfficientNetV2S(input_shape=(384, 384, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    return EfficientNetV2(model_type="s", model_name="EfficientNetV2S", **locals(), **kwargs)


def EfficientNetV2M(input_shape=(480, 480, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    return EfficientNetV2(model_type="m", model_name="EfficientNetV2M", **locals(), **kwargs)


def EfficientNetV2L(input_shape=(480, 480, 3), num_classes=1000, dropout=0.4, classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    return EfficientNetV2(model_type="l", model_name="EfficientNetV2L", **locals(), **kwargs)


def EfficientNetV2XL(input_shape=(512, 512, 3), num_classes=1000, dropout=0.4, classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    return EfficientNetV2(model_type="xl", model_name="EfficientNetV2XL", **locals(), **kwargs)


def get_actual_survival_probabilities(model):
    from tensorflow_addons.layers import StochasticDepth

    return [ii.survival_probability for ii in model.layers if isinstance(ii, StochasticDepth)]
