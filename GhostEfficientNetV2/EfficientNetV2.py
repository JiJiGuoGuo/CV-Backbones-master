import tensorflow as tf
import math

def hard_swish(x, inplace: bool = False):
    '''
    比swish更简洁省时，通常h_swish通常只在更深层次上有用
    '''
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return (tf.nn.relu6(x + 3.) * x)/ 6.

def hard_sigmoid(x):
    '''
    hard_sigmoid是Logistic sigmoid的分段近似函数，更易于计算，学习速率加快
    if x<-2.5,return 0
    if x>2.5,return 1
    if -2.5<=x<=2.5,return 0.2*x+0.5
    tensorflow2已经实现了hard_sigmoid
    '''
    return tf.keras.activations.hard_sigmoid(x)

def _make_divisible(v, divisor:int=8, min_value=None):
    """
    确保所有的层的通道都能够被8整除，生成的数字，能被divisor整除
    使用if，来保证new_v相对于原先的v的变化不超过+-10%
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    # 确保这一整数的下降幅度不超过10%。
    if new_v < (0.9 * v):
        new_v += divisor
    return new_v

class SE(tf.keras.layers.Layer):
    def __init__(self,inputs_channels:int,se_ratio:int = 0.25,name:str=""):
        '''
        这个函数是使用Conv1x1实现的SE模块，并使用reduc_mean实现GlobalAveragePooling
        Args:
            inputs_channels: 输入张量的channels
            se_ratio: 第一个FC会将输入SE的张量channels压缩成的倍率
            name:
            return:一个张量，shape同input
        '''
        super(SE,self).__init__()
        self.se_ratio = se_ratio
        self.filters = inputs_channels
        self.reduction = _make_divisible(inputs_channels * se_ratio,8)
        #第一个FC将输入SE的channel压缩成1/4
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')
        self.conv1 = tf.keras.layers.Conv2D(self.reduction,1,1,use_bias=True,name=name+'1_conv')
        self.act1 = tf.keras.layers.Activation('swish')
        self.conv2 = tf.keras.layers.Conv2D(self.filters,1,1,use_bias=True,name=name+'2_conv')
        self.act2 = tf.keras.layers.Activation('sigmoid')
        self.multiply = tf.keras.layers.Multiply()
    def call(self,inputs):
        #由于tf2.6才增加了keep_dim，所以在tf2.3需要手动expand_dims
        x = self.global_pool(inputs)
        x = tf.expand_dims(tf.expand_dims(x,1),1)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        out = self.multiply([x,inputs])
        return out

class SqueezeExcite(tf.keras.layers.Layer):
    def __init__(self, input, se_ratio=0.25, reduced_base_chs=None, divisor=4):
        '''
        SE模块，Squeeze：一个AvePooling，GhostModue的SE全部激活函数换成了relu
        Excitation：一个FC+swish，加一个FC+sigmoid
        Args:
            input:
            se_ratio: 第一个FC会将输入SE的张量channels压缩成的倍率
            reduced_base_chs:
            divisor:
            return:一个张量，shape同input
        '''
        super(SqueezeExcite, self).__init__()
        self.input_channels = input.shape[-1]
        reduced_chs = _make_divisible((reduced_base_chs or self.input_channels) * se_ratio, divisor)
        self.conv1 = tf.keras.layers.Conv2D(reduced_chs,1,1,use_bias=True)
        self.act1 = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv2D(self.input_channels,1,1,use_bias=True)#使得输出channels=输入chanels
        self.act2 = tf.keras.layers.Activation('relu')

    def call(self, x):
        xx = tf.reduce_mean(x,(1,2),keepdims=True)
        xx = self.conv1(xx)
        xx = self.act1(xx)
        xx = self.conv2(xx)
        xx = self.act2(xx)
        out = tf.keras.layers.Multiply()([x,xx])
        return out

class GhostModule(tf.keras.layers.Layer):
    def __init__(self, input_channels,kernel_size=1, ratio=2, dw_size=3, stride=1,use_relu=True):
        '''
        实现的GhostModule，CNN模型的中的feature map的很多是相似的，某些origin map能够通过某种cheap operation
        生成这些相似的feature map，称为ghost map，中文翻译为幻影
        Args:
            input_channels: 输入的张量的通道数
            kernel_size: 除1x1卷积的其他卷积核大小
            ratio:初始conv会将原channel压缩成原来的多少
            dw_size: DepthwiseConv的卷积核大小
            stride:
            use_relu: 是否使用relu作为激活函数
            return:GhostModule不改变input的shape，所以输入channels=输出channels
        '''
        super(GhostModule, self).__init__()
        self.ouput_channel = input_channels
        init_channels = math.ceil(self.ouput_channel / ratio)
        new_channels = init_channels * (ratio - 1)

        # 这里可以看到实现了两次卷积，点卷积
        self.primary_conv = tf.keras.Sequential([
            #点卷积的卷积核的组数=上一层的channel数，大小为1x1xM，其中M=input.shape(-1)
            tf.keras.layers.Conv2D(init_channels,kernel_size,stride,padding='same',use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1e-5),
            tf.keras.layers.Activation(activation='relu') if use_relu else tf.keras.Sequential(),
        ])

        self.cheap_operation = tf.keras.Sequential([
            #group用于对channel进行分组，默认是一个channel为一组,这里采用的是分组卷积
            tf.keras.layers.Conv2D(new_channels,3,1,'same',use_bias=False,groups=init_channels),
            # tf.keras.layers.DepthwiseConv2D(3,1,'same',use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1e-5),
            tf.keras.layers.Activation(activation='relu') if use_relu else tf.keras.Sequential(),
        ])

    def call(self,x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = tf.concat([x1,x2],axis=-1)#origin map和ghost map进行拼接
        #第0,1,2维全选，最后一维度从0开始读取到self.oup，步长为1，左闭右开
        return out[...,:self.ouput_channel]

class Fused_MBConv(tf.keras.layers.Layer):
    def __init__(self,input_channels,output_channels,kernel_size,activation='swish',
                 stride=1,expand_ratio=6,se_ratio=4,dropout=None,shortcut = 1,survival=None,epsilon=1e-5):
        super(Fused_MBConv, self).__init__()
        self.expand_ratio = expand_ratio
        self.drop = dropout
        self.se_ratio = se_ratio
        self.use_shortcut = shortcut
        self.survival = survival
        expand_ratio_filters = _make_divisible(input_channels * expand_ratio)
        self.stride = stride
        self.input_channels = input_channels
        self.output_channels = output_channels

        if stride == 2:
            self.poolAvage = tf.keras.layers.AveragePooling2D()
        if input_channels != output_channels:
            self.shortcut = tf.keras.layers.Conv2D(output_channels,kernel_size=1,strides=1,padding='same',use_bias = False)

        #升维阶段，卷积
        if expand_ratio != 1:
            self.conv3x3_fused = tf.keras.Sequential([
                tf.keras.layers.Conv2D(expand_ratio_filters,kernel_size=kernel_size,strides=stride,padding='same',use_bias=False),
                tf.keras.layers.BatchNormalization(epsilon=epsilon),
                tf.keras.layers.Activation(activation),
            ])
            if (dropout is not None) and (dropout != 0):
                self.ghost1_dropout = tf.keras.layers.Dropout(dropout)

        #se模块
        if se_ratio is not None:
            self.se = SE(expand_ratio_filters, se_ratio)

        #输出阶段，降维阶段，卷积
        self.conv1x1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(output_channels,kernel_size=1 if expand_ratio != 1 else kernel_size,
                                   strides=1 if expand_ratio != 1 else stride,padding='same',
                                   use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=epsilon),
        ])

    def call(self,inputs):
        shortcut = inputs
        if self.stride == 2:
            shortcut = self.poolAvage(shortcut)
        if self.input_channels != self.output_channels:
            shortcut = self.shortcut(shortcut)
        #升维
        if self.expand_ratio != 1:
            inputs = self.conv3x3_fused(inputs)
            if (self.drop is not None) and (self.drop != 0):
                inputs = self.ghost1_dropout(inputs)
        #SE模块
        if self.se_ratio is not None:
            inputs = self.se(inputs)
        #输出阶段，降维1x1
        inputs = self.conv1x1(inputs)

        if self.use_shortcut:#如果使用直连/残差结构
            if self.survival is not None and self.survival<1:#生存概率(随机深度残差网络论文中的术语，表示残差支路被激活的概率)
                from tensorflow_addons.layers import StochasticDepth
                stoDepth = StochasticDepth(survival_probability=self.survival)
                return stoDepth([shortcut, inputs])
            else:
                return tf.keras.layers.Add()([inputs,shortcut])
        else:
            return inputs

class MBConv(tf.keras.layers.Layer):
    def __init__(self,input_channels,output_channels,kernel_size,activation='swish',
                 stride=1,expand_ratio=6,se_ratio=4,dropout=None,shortcut = 1,survival=None,epsilon=1e-5):
        super(MBConv, self).__init__()
        expand_channels = expand_ratio * input_channels
        self.expand_ratio = expand_ratio
        self.dropout = dropout
        self.survival = survival
        self.use_shortcut = shortcut
        self.stride = stride
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.has_se = (se_ratio is not None) and (0 < se_ratio <=1)#如果有se_ratio则表示要使用se模块

        if stride == 2:
            self.poolAvage = tf.keras.layers.AveragePooling2D()
        if input_channels != output_channels:
            self.shortcut = tf.keras.layers.Conv2D(output_channels,kernel_size=1,strides=1,padding='same',use_bias = False)

        #conv1x1升维
        if expand_ratio != 1:#pytorch没有这个expand=1的判断，难道是1x1在输出=原始维度时，没有什么效果
            self.conv1x1_up = tf.keras.Sequential([
                tf.keras.layers.Conv2D(expand_channels,kernel_size=1,strides=1,padding='same',use_bias=False),
                tf.keras.layers.BatchNormalization(epsilon=epsilon),
                tf.keras.layers.Activation(activation),
            ])
        #depthwise3x3
        self.dethwise_conv = tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,strides=stride,padding='same',use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=epsilon),
            tf.keras.layers.Activation(activation),
        ])

        #是否dropout
        if (expand_ratio != 1) and (dropout is not None) and (dropout != 0):
            self.dropout = tf.keras.layers.Dropout(dropout)
        #SE模块
        if self.has_se:
            self.se = SE(expand_channels, se_ratio)
        #conv1x1降维
        self.conv1x1_down = tf.keras.Sequential([
            tf.keras.layers.Conv2D(output_channels,kernel_size=1,strides=1,use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=epsilon),
        ])

    def call(self,inputs):
        shortcut = inputs
        if self.stride == 2:
            shortcut = self.poolAvage(shortcut)
        if self.input_channels != self.output_channels:
            shortcut = self.shortcut(shortcut)

        if self.expand_ratio != 1:#conv1x1升维
            inputs = self.conv1x1_up(inputs)
        #depthwise3x3
        inputs = self.dethwise_conv(inputs)
        #dropout
        if (self.expand_ratio != 1) and (self.dropout is not None) and (self.dropout != 0):
            x = self.dropout(inputs)
        #se模块
        if self.has_se:
            inputs = self.se(inputs)
        #conv1x1降维
        inputs = self.conv1x1_down(inputs)

        #shortcut and stochastic Depth
        if self.use_shortcut:#如果使用直连/残差结构
            if self.survival is not None and self.survival<1:#生存概率(随机深度残差网络论文中的术语，表示残差支路被激活的概率)
                from tensorflow_addons.layers import StochasticDepth
                stoDepth = StochasticDepth(survival_probability=self.survival)
                return stoDepth([shortcut,inputs])
            else:
                return tf.keras.layers.Add()([inputs,shortcut])
        else:
            return inputs


class EfficientNetV2(tf.keras.Model):
    '''
    根据EfficientNetV2论文重新实现的EfficientNet-V2-s和官方代码
    Args:
        cfg: stages的配置
        num_classes: 类别数量，也是最终的输出channels
        input: 输入的张量, 若提供了则忽略in_shape
        activation: 通过隐藏层的激活函数
        width_mult: 模型宽度因子, 默认为1
        depth_mult: 模型深度因子,默认为1
        conv_dropout_rate: 在MBConv/Stage后的drop的概率，0或none代表不使用dropout
        dropout_rate: 在GlobalAveragePooling后的drop概率，0或none代表不使用dropout
        drop_connect: 在跳层连接drop概率，0或none代表不使用dropout
        include_top: 是否包含分类用的最顶层
        name: 层的名字
    Returns:a tf.keras model
    '''
    def __init__(self,cfg,num_classes,activation,width_mult:float,depth_mult=float,
                 conv_dropout_rate=None,dropout_rate=None,drop_connect=None,include_top=True,name=None,epsilon=1e-5):
        super(EfficientNetV2, self).__init__(name=name)
        self.dropout_rate = dropout_rate
        self.include_top = include_top
        #stage 0
        self.stage0_conv3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(24,kernel_size=3,strides=2,padding='same',use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1e-5),
            tf.keras.layers.Activation(activation),
        ])
        #接下来是stage 1到stage 6
        self.stage1to6 = tf.keras.Sequential()
        for stage in cfg:
            count = int(math.ceil((stage[0] * depth_mult)))#stage[0]是count表示重复多少次
            for j in range(count):
                self.stage1to6.add(handleInputStageChannels(index=j,input_channels=_make_divisible(stage[4],width_mult),
                                                            output_channels=_make_divisible(stage[5],width_mult),
                                                            kernel_size=stage[1],activation=activation,expand_ratio=stage[3],
                                                            use_Fused=stage[6],stride=stage[2],se_ratio=stage[7],dropout=conv_dropout_rate,
                                                            drop_connect=drop_connect,shortcut=stage[8],survival=stage[9]))
        #最终stage
        self.stage7_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(_make_divisible(1280,width_mult),kernel_size=1,padding='same',use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=epsilon),
            tf.keras.layers.Activation(activation),
        ])
        self.stage7_globalAverPool = tf.keras.layers.GlobalAveragePooling2D()
        if (self.dropout_rate is not None) and (self.dropout_rate != 0):
            self.stage7_drop = tf.keras.layers.Dropout(dropout_rate)

        self.stage7_classfier = tf.keras.Sequential([
            tf.keras.layers.Dense(num_classes),
            # tf.keras.layers.Activation('softmax'),
        ])
    def call(self,inputs):
        x = self.stage0_conv3(inputs)
        x = self.stage1to6(x)
        x = self.stage7_conv(x)
        x = self.stage7_globalAverPool(x)
        if (self.dropout_rate is not None) and (self.dropout_rate != 0):
            x = self.stage7_drop(x)
        x = self.stage7_classfier(x)
        return x

def handleInputStageChannels(index,input_channels,output_channels,kernel_size,activation,expand_ratio,use_Fused,
                             stride=1,se_ratio=None,dropout=None,drop_connect=0.2,shortcut=1,survival=None):
    '''
    这个函数用来处理在循环count时，在每组count的第一个stage到第二个stage的channels切换，导致的stage输入问题的情况
    Args:
        count: 总的重复次数
        input_channels:
        output_channels:
        kernel_size:
        activation:
        expand_ratio:
        use_Fused:
        stride:
        se_ratio:
        dropout:
        drop_connect:
    Returns:
    '''
    if use_Fused:
        return Fused_MBConv(input_channels = output_channels if index != 0 else input_channels,
                                  output_channels = output_channels,kernel_size=kernel_size,activation=activation,
                                  stride = 1 if index != 0 else stride,
                                  expand_ratio = expand_ratio,se_ratio=se_ratio,dropout=dropout,shortcut=shortcut,survival=survival)
    elif not use_Fused:
        return MBConv(input_channels = output_channels if index != 0 else input_channels,
                                  output_channels = output_channels,kernel_size=kernel_size,activation=activation,
                                  stride = 1 if index != 0 else stride,
                                  expand_ratio = expand_ratio,se_ratio=se_ratio,dropout=dropout,shortcut=shortcut,survival=survival)


class EfficientNetV2_S(tf.keras.Model):
    def __init__(self,num_classes,activation='swish',width_mult=1.0,depth_mult=1.0,conv_dropout_rate=None,dropout_rate=None,drop_connect=0.2):
        super(EfficientNetV2_S, self).__init__()
        # 计数：该stage重复多少次；扩展比例：MBConv第一个卷积将输入通道扩展成几倍(1,4,6)；SE率：SE模块中第一个FC/Conv层将其缩放到多少，通常是1/4
        # 次数0，卷积核大小1，步长2，扩展比例3，输入通道数4，输出通道数5，是否Fused6，SE率7，是否shortcut8,生存概率9
        #   0,  1  2, 3  4   5   6       7   8  9
        cfg = [
            [2, 3, 1, 1, 24, 24, True, None, 1, 0.5],  # stage 1
            [4, 3, 2, 4, 24, 48, True, None, 1, 0.5],  # stage 2
            [4, 3, 2, 4, 48, 64, True, None, 1, 0.5],  # stage 3
            [6, 3, 2, 4, 64, 128, False, 4, 1, 0.5],  # stage 4
            [9, 3, 1, 6, 128, 160, False, 4, 1, 0.5],  # stage 5
            [15, 3, 2, 6, 160, 256, False, 4, 1, 0.5],  # stage 6
        ]
        self.efficientV2 = EfficientNetV2(cfg,num_classes=num_classes,activation=activation,
                                          width_mult=width_mult,depth_mult=depth_mult,
                                          conv_dropout_rate=None,dropout_rate=None,drop_connect=None)
    def call(self,inputs):
        return self.efficientV2(inputs)

if __name__ == '__main__':
    x = tf.random.uniform([3,224,224,3])
    model = EfficientNetV2_S(7)
    model(x)
    model.summary()
