import tensorflow as tf
import math
import numpy as np

def hard_swish(x, inplace: bool = False):
    '''
    比swish更简洁省时，通常h_swish通常只在更深层次上有用
    '''
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return (tf.nn.relu6(x + 3.) * x)/ 6.

def hard_sigmoid():
    '''
    hard_sigmoid是Logistic sigmoid的分段近似函数，更易于计算，学习速率加快
    if x<-2.5,return 0
    if x>2.5,return 1
    if -2.5<=x<=2.5,return 0.2*x+0.5
    tensorflow2已经实现了hard_sigmoid
    '''
    return tf.keras.activations.hard_sigmoid()

def round_filter(filters,multiplier=1.0):
    divisor = 8
    min_depth = 8
    filters = filters * multiplier
    new_filters = max(min_depth,int(filters + divisor/2)//divisor * divisor)
    return new_filters


class SE(tf.keras.Model):
    def __init__(self,inputs_channels:int,se_ratio:int = 4):
        '''
        这个函数是使用Conv1x1实现的SE模块，并使用reduc_mean实现GlobalAveragePooling
        Args:
            inputs_channels: 输入张量的channels
            se_ratio: 第一个FC会将输入SE的张量channels压缩成的倍率
            name:
            return:一个张量，shape同input
        '''
        super(SE,self).__init__()
        self.filters = inputs_channels
        self.reducation = round_filter(inputs_channels / se_ratio)
        self._build()
    def _build(self):
        #squeeze
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')
        #第一个FC将输入SE的channel压缩成1/4,excitation
        self.excit = tf.keras.Sequential([
            # tf.keras.layers.Conv2D(self.reducation,1,1,padding='same',use_bias=True),
            # tf.keras.layers.Activation('relu'),
            # tf.keras.layers.Conv2D(self.filters, 1, 1, padding='same',use_bias=True),
            # tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.Dense(self.reducation),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(self.filters),
            tf.keras.layers.Activation('relu'),
        ])
        self.multiply = tf.keras.layers.Multiply()
    def call(self,inputs):
        #由于tf2.6才增加了keep_dim，所以在tf2.3需要手动expand_dims
        x = self.global_pool(inputs)
        x = tf.expand_dims(tf.expand_dims(x,1),1)
        x = self.excit(x)
        out = self.multiply([x,inputs])
        return out

class GhostModule(tf.keras.layers.Layer):
    def __init__(self, input_channels,kernel_size=1, ratio=2, stride=1,activation='elu',epsilon=1e-5):
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
            tf.keras.layers.BatchNormalization(epsilon=epsilon),
            tf.keras.layers.Activation(activation=activation),
        ])

        self.cheap_operation = tf.keras.Sequential([
            #group用于对channel进行分组，默认是一个channel为一组,这里采用的是分组卷积
            # tf.keras.layers.Conv2D(new_channels,3,1,'same',use_bias=False,groups=init_channels),
            tf.keras.layers.DepthwiseConv2D(3,1,'same',use_bias=False),
        ])

    def call(self,x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = tf.concat([x1,x2],axis=-1)#origin map和ghost map进行拼接
        #第0,1,2维全选，最后一维度从0开始读取到self.oup，步长为1，左闭右开
        return out

class Ghost_Fused_MBConv(tf.keras.layers.Layer):
    def __init__(self,input_channels,output_channels,kernel_size,activation,
                 stride=1,expand_ratio=6,se_ratio=4,dropout=None,shortcut = 1,survival=None,epsilon=1e-5):
        super(Ghost_Fused_MBConv, self).__init__()
        self.expand_ratio = expand_ratio
        self.drop = dropout
        self.se_ratio = se_ratio
        self.use_shortcut = shortcut
        self.survival = survival
        expand_ratio_filters = round_filter(input_channels * expand_ratio)
        self.stride = stride
        self.input_channels = input_channels
        self.output_channels = output_channels

        #用于校正通道
        if stride == 2:
            self.poolAvage = tf.keras.layers.AveragePooling2D()
        if input_channels != output_channels:
            self.shortcut = GhostModule(output_channels,kernel_size=1,stride=1,activation=activation)

        #升维阶段，卷积
        if expand_ratio != 1:
            self.ghost1 = GhostModule(expand_ratio_filters,
                                      kernel_size=kernel_size,stride=stride,ratio=2,activation=activation)
            self.ghost1_bn = tf.keras.layers.BatchNormalization(epsilon=epsilon)
            self.ghost1_act = tf.keras.layers.Activation(activation)
            if (dropout is not None) and (dropout != 0):
                self.ghost1_dropout = tf.keras.layers.Dropout(dropout)

        #se模块
        if se_ratio is not None:
            self.se = SE(expand_ratio_filters, se_ratio)

        #输出阶段，降维阶段，卷积
        self.ghost2 = GhostModule(output_channels,kernel_size=1 if expand_ratio != 1 else kernel_size,
                                  stride=1 if expand_ratio != 1 else stride,activation=activation)
        self.out_bn = tf.keras.layers.BatchNormalization(epsilon=epsilon)

    def call(self,inputs):
        shortcut = inputs
        if self.stride == 2:
            shortcut = self.poolAvage(shortcut)
        if self.input_channels != self.output_channels:
            shortcut = self.shortcut(shortcut)
        #升维
        if self.expand_ratio != 1:
            inputs = self.ghost1(inputs)
            inputs = self.ghost1_bn(inputs)
            inputs = self.ghost1_act(inputs)
            if (self.drop is not None) and (self.drop != 0):
                inputs = self.ghost1_dropout(inputs)
        #SE模块
        if self.se_ratio is not None:
            inputs = self.se(inputs)

        inputs = self.ghost2(inputs)
        inputs = self.out_bn(inputs)

        if self.use_shortcut:#如果使用直连/残差结构
            if self.survival is not None and self.survival<1:#生存概率(随机深度残差网络论文中的术语，表示残差支路被激活的概率)
                from tensorflow_addons.layers import StochasticDepth
                stoDepth = StochasticDepth(survival_probability=self.survival)
                return stoDepth([shortcut, inputs])
            else:
                return tf.keras.layers.Add()([inputs,shortcut])
        else:
            return inputs

class Ghost_MBConv(tf.keras.layers.Layer):
    def __init__(self,input_channels,output_channels,kernel_size,activation,
                 stride=1,expand_ratio=6,se_ratio=4,dropout=None,shortcut = 1,survival=None,epsilon=1e-5):
        super(Ghost_MBConv, self).__init__()
        expand_channels = expand_ratio * input_channels
        self.expand_ratio = expand_ratio
        self.dropout = dropout
        self.se_ratio = se_ratio
        self.survival = survival
        self.use_shortcut = shortcut
        self.stride = stride
        self.input_channels = input_channels
        self.output_channels = output_channels

        if stride == 2:
            self.poolAvage = tf.keras.layers.AveragePooling2D()
        if input_channels != output_channels:
            self.shortcut = GhostModule(output_channels,kernel_size=1,stride=1,activation=activation)
        #conv1x1
        if expand_ratio != 1:
            self.ghost1 = GhostModule(expand_channels,kernel_size=1,ratio=2,stride=1,activation=activation)
            self.ghost1_bn = tf.keras.layers.BatchNormalization(epsilon=1e-5)
            self.ghost1_act = tf.keras.layers.Activation(activation)
        #depthwise3x3
        self.dethwise = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,strides=stride,
                                                        padding='same',use_bias=False)
        self.dethwise_bn = tf.keras.layers.BatchNormalization(epsilon=1e-5)
        self.dethwise_act = tf.keras.layers.Activation(activation)
        #是否dropout
        if (expand_ratio != 1) and (dropout is not None) and (dropout != 0):
            self.dropout = tf.keras.layers.Dropout(dropout)
        #SE模块
        if se_ratio is not None:
            self.se = SE(expand_channels, se_ratio)
        #conv1x1
        self.ghost2 = GhostModule(output_channels,kernel_size=1,stride=1,activation=activation)
        self.ghost2_bn = tf.keras.layers.BatchNormalization(epsilon=epsilon)

    def call(self,inputs):
        shortcut = inputs
        if self.stride == 2:
            shortcut = self.poolAvage(shortcut)
        if self.input_channels != self.output_channels:
            shortcut = self.shortcut(shortcut)

        if self.expand_ratio != 1:#conv1x1
            inputs = self.ghost1(inputs)
            inputs = self.ghost1_bn(inputs)
            inputs = self.ghost1_act(inputs)
        #depthwise3x3
        inputs = self.dethwise(inputs)
        inputs = self.dethwise_bn(inputs)
        inputs = self.dethwise_act(inputs)
        #dropout
        if (self.expand_ratio != 1) and (self.dropout is not None) and (self.dropout != 0):
            x = self.dropout(inputs)
        #se模块
        if self.se_ratio is not None:
            inputs = self.se(inputs)
        #conv1x1
        inputs = self.ghost2(inputs)
        inputs = self.ghost2_bn(inputs)
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

class GhostBottleneck(tf.keras.layers.Layer):
    '''
    G-Bneck专为小型网络而设计类似于ResidualBlock，包含两个GhostModule，第一个用作扩展层，第二个减少通道数
    stride=1和2的情形不同
    '''
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,stride=1, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.#是否使用SEModule
        self.stride = stride

        #第一个GhostModule用于扩展层
        self.ghost_module1 = GhostModule(in_chs, mid_chs)

        #深度卷积Depthwise Convolution
        if self.stride > 1:#大于2才使用DepthwiseConv
            self.dw_conv = tf.keras.layers.DepthwiseConv2D(dw_kernel_size,strides=2,groups=mid_chs,padding='same',use_bias=False)
            self.dw_bn = tf.keras.layers.BatchNormalization()

        if has_se:#如果使用SE模块
            self.se = SE(mid_chs, se_ratio=se_ratio)
        else: #不使用SE模块
            self.se = None

        #第二个GhostModule模块不再使用ReLU，减少通道数
        self.ghost_module2 = GhostModule(mid_chs, out_chs, use_relu=False)

        #是否使用shortcut
        #先假设规定：要使用shortcut，GBneck的输入特征图channels != 输出的特征图channels
        #因为如果input_channel = out_channels那么就不能使用shortcut直连
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = tf.keras.Sequential()
        else:#使用shotcut，stride>2
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.DepthwiseConv2D(dw_kernel_size,stride,padding='same',use_bias=False,groups=in_chs),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(out_chs,1,1,padding='same',use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])

    def call(self, x):

        #第一个GhostModule
        xx = self.ghost_module1(x)

        # Depth-wise convolution
        if self.stride > 1:
            xx = self.dw_conv(xx)
            xx = self.dw_bn(xx)

        #SE模块
        if self.se is not None:
            xx = self.se(xx)
        #第二个GhostModule
        xx = self.ghost_module2(xx)
        #残差结构
        xx =xx + self.shortcut(x)
        return xx

class EfficientNetV2(tf.keras.layers.Layer):
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
    Returns:a tf.keras model
    '''
    def __init__(self,cfg,num_classes:float,activation:str,width_mult:float,depth_mult:float,
                 conv_dropout_rate=None,dropout_rate=None,drop_connect=None,epsilon=1e-5):
        super(EfficientNetV2, self).__init__()
        self.dropout_rate = dropout_rate
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
                self.stage1to6.add(handleInputStageChannels(index=j,input_channels=round_filter(stage[4],width_mult),
                                                            output_channels=round_filter(stage[5],width_mult),
                                                            kernel_size=stage[1],activation=activation,expand_ratio=stage[3],
                                                            use_Fused=stage[6],stride=stage[2],se_ratio=stage[7],dropout=conv_dropout_rate,
                                                            drop_connect=drop_connect,shortcut=stage[8],survival=stage[9]))
        #最终stage
        self.stage7_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(round_filter(1280,width_mult),kernel_size=1,padding='same',use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1e-5),
            tf.keras.layers.Activation(activation),
        ])
        self.stage7_globalAverPool = tf.keras.layers.GlobalAveragePooling2D()
        if (self.dropout_rate is not None) and (self.dropout_rate != 0):
            self.stage7_drop = tf.keras.layers.Dropout(dropout_rate)

        self.stage7_classfier = tf.keras.Sequential([
            tf.keras.layers.Dense(num_classes),
        ])

    def get_grim_top_right(self, matrix):
        '''
        求一个矩阵的右上角元素矩阵
        :param matrix: 传入一个numpy，类型为[batch,height*width,channel]
        :return: 返回右上角元素组成的numpy数组. [[1,2,3,4...]]
        '''
        batch_size = matrix.shape[0]
        # 这里需要变成[batch,height*width,channel]
        matrix = np.array(matrix)
        features = []
        for index in range(batch_size):
            gram_matrix = np.dot(matrix[index].T, matrix[index])
            feature = []
            gram_len = gram_matrix.shape[1]
            for row in range(gram_len):
                tem_row = []
                for clo in range(gram_len):
                    clos = clo + row
                    if (clos > gram_len - 1):
                        break
                    tem_row.append(gram_matrix[row][row + clo])
                #             print(tem_row)
                feature.append(tem_row)
            features.append(feature)
        return np.array(features)

    def call(self,inputs):
        x = self.stage0_conv3(inputs)#stage0
        x = self.get_grim_top_right(x)

        x = self.stage1to6(x)#stage1-6
        x = self.stage7_conv(x)#stage7
        x = self.stage7_globalAverPool(x)#stage8
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
        return Ghost_Fused_MBConv(input_channels = output_channels if index != 0 else input_channels,
                                  output_channels = output_channels,kernel_size=kernel_size,activation=activation,
                                  stride = 1 if index != 0 else stride,
                                  expand_ratio = expand_ratio,se_ratio=se_ratio,dropout=dropout,shortcut=shortcut,survival=survival)
    elif not use_Fused:
        return Ghost_MBConv(input_channels = output_channels if index != 0 else input_channels,
                                  output_channels = output_channels,kernel_size=kernel_size,activation=activation,
                                  stride = 1 if index != 0 else stride,
                                  expand_ratio = expand_ratio,se_ratio=se_ratio,dropout=dropout,shortcut=shortcut,survival=survival)

class EfficientNetV2_S(tf.keras.Model):
    def __init__(self,num_classes,activation,width_mult=1.0,depth_mult=1.0,conv_dropout_rate=None,dropout_rate=None,drop_connect=0.2):
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
    model = EfficientNetV2_S(7,activation='relu',drop_connect=0.2,conv_dropout_rate=0.2,dropout_rate=0.2)
    model(x)
    model.summary()
