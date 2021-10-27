#这个文件是作者的模型文件，#Gram Matrix能够反映该组向量中，各个向量之间的某种关系，self用来指向类的实例对象，而不是类本身
#tf.keras.layers.Layer是一个可调用对象，以张量作为输入和输出，计算方法定义在call()里，官方建议定义__init__()，build(self,input_hape)
#call(),get_config()
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import (Dense,Dropout,LayerNormalization,)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd


#----------------------------------model-----------------------------------
#多头注意力机制
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):#当创建类，就会调用该方法，self代表类的实例，不必传入实际的参数
        '''
        用于初始化
        :param embed_dim:词向量或字向量维度
        :param num_heads: 多头注意力机制的头数量
        '''
        super(MultiHeadSelfAttention, self).__init__()#主要用来调用父类的方法，调用了父类的初始化
        self.embed_dim = embed_dim#64
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                #python使用f表示在字符串内支持大括号的python表达式，大括号内会替换成对应的变量值
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        #创建3个qkv层
        self.query_dense = Dense(embed_dim)#输出维度是64
        self.key_dense = Dense(embed_dim)#输出维度64
        self.value_dense = Dense(embed_dim)#输出维度64
        self.combine_heads = Dense(embed_dim)#输出维度64

    #根据Q，K，V矩阵计算注意力得分
    def attention(self, query, key, value):
        '''
        用于计算qkv三个矩阵的注意力得分，由于qkv三个矩阵都需要使用WqWkWv的权重矩阵获得
        需要提供其全连接层的训练方法。
        :param query: 待查序列
        :param key: 关键字序列
        :param value: 值序列
        :return: qkv的注意力得分和公式左部分softmax部分的值
        '''
        #一个矩阵乘法，transpose_b表示是否将处于b位置matrix转置，可选参数可以对稀疏矩阵进行优化计算
        score = tf.matmul(a=query, b=key,transpose_b=True,b_is_sparse=False)
        #可以理解为取最后一维的元素数量,-1表示从最后开始取,取出来的是一个0维张量（一个数）
        dim_key = tf.cast(key.shape[-1], tf.float32)
        scaled_score = score / tf.sqrt(dim_key)        #除以一个数
        weights = tf.nn.softmax(scaled_score, axis=-1)        #softmax转换，公式左部分
        output = tf.matmul(weights, value) #矩阵乘法，乘V矩阵，公式最终结果
        return output, weights

    def separate_heads(self, x, batch_size):
        '''
        x原始的词向量维度为embed_dim，将其除以num_head,也即
        self.projection_dim=embde_dim//num_heads
        :param x: 需要处理的参数
        :param batch_size: 一个batch大小
        :return: 返回被处理过后的x
        '''
        #重塑shape，不改变元素顺序,原本的一个词向量是embed_dim维度，加了多个头后，就将embed_dim划分为num_heads个。
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.embed_dim // self.num_heads])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        '''
        call用来定义Model的计算方法，这也是tf.keras.layer.Layer需要定义的计算方法
        用来指定训练和推理过程中的行为，当定义了类的实例对象，并且传入参数时，call就会被调用
        :param inputs: 需要处理的输入
        :return:多头注意力机制部分的输出结果
        '''
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)        #注意力层的输出结果，左部分
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(tf.keras.layers.Layer):#TransformerBlock就是Encoder主要由位置编码和多头注意力机制
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()#super调用父类的方法，解决多重继承的问题
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = tf.keras.Sequential(#Sequential容器用来将多个网络层封装成一个大型的网络
            [
                Dense(mlp_dim, activation=tfa.activations.gelu),#损失函数为高斯误差线性单元
                Dropout(dropout),
                Dense(embed_dim),
                Dropout(dropout),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)#spsilon添加到方差中，以免被0除
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training):#call用来指定训练和推理中的不同行为
        print("inputs到TB的shape,位于ln之前{}".format(inputs.shape))
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm)
        print("inputs到TB的shape,位于attention之后{}".format(inputs.shape))
        attn_output = self.dropout1(attn_output, training=training)#traing=true表示本层在训练和推理才使用
        out1 = attn_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        print("inputs到TB的shape,位于TBSequential之后{}".format(inputs.shape))
        mlp_output = self.dropout2(mlp_output, training=training)
        return mlp_output + out1


# Model将layers组成了一个可训练和推理的对象
class VisionTransformer(tf.keras.Model):
    def __init__(self, image_size, patch_size, num_layers, num_classes, d_model, num_heads, mlp_dim, channels, dropout):
        '''对于网络层的定义放在这里'''
        # super使用来调用父类的一个方法
        super(VisionTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2  # patches的数量，48/4=144
        self.patch_dim = channels * patch_size ** 2  # 每个patch打平后的向量维度

        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers

        # Rescaling是一个类
        self.rescale = Rescaling(scale=1.0 / 255)  # 缩放，参数是缩放比例，将原来比例是[0,255]缩放到[0,1]

        # add_weight()用来创建不依赖于input shapes的层状态变量，加入一个变量到层
        self.pos_emb = self.add_weight(name="pos_emb", shape=(1, num_patches + 1, d_model))
        self.class_emb = self.add_weight(name="class_emb", shape=(1, 1, d_model))

        # Dense创建一个Dense层，权重为W(mxn)，输入n(nx1)，激活函数Activation，偏置b(nx1)，out=Activation(Wx+b)
        self.patch_proj = Dense(units=d_model)
        self.enc_layers = [TransformerBlock(embed_dim=d_model, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)
                           for _ in range(num_layers)]
        #         self.mlp_head = tf.keras.Sequential(#创建一个连续的模型实例
        #             [
        #                 LayerNormalization(epsilon=1e-6),
        #                 Dense(mlp_dim, activation=tfa.activations.gelu),
        #                 Dropout(dropout),
        #                 Dense(num_classes),
        #             ]
        #         )
        self.mlp_ln = LayerNormalization(epsilon=1e-6)
        self.mlp_dense1 = Dense(mlp_dim, activation=tfa.activations.gelu)
        self.mlp_drop = Dropout(dropout)
        self.mlp_dense2 = Dense(num_classes,name='mlp_head_dense2')

    def extract_patches(self, images):  # 提取patch
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(  # 将图片分割为多个子图片,并且返回一个4D张量，与images一样
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],  # 分割成的图片大小
            strides=[1, self.patch_size, self.patch_size, 1],  # 移动步长，这里步长和patch_size是相同的所以，没有overlap
            rates=[1, 1, 1, 1],  # 表示每几个像素点取一个像素点，这里表示每个像素都要提取
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def get_grim_top_right(self,matrix):
        '''
        求一个矩阵的右上角元素矩阵
        :param matrix: 传入一个tensor，类型为[batch,height,width,channel]
        :return: 返回格拉姆右上角元素组成的numpy数组. [batch,height,width,channel]
        '''
        matrix_transpose = tf.transpose(matrix,[0,3,1,2])
        grim_matrix = tf.matmul(matrix_transpose,matrix_transpose,transpose_a=True)
        upper_matrix = tf.linalg.band_part(grim_matrix,0,-1)
        matrix_transposed = tf.transpose(upper_matrix,[0,2,3,1])
        return matrix_transposed


    def call(self, x, training):
        '''
        call用来指定训练和推理中的不同行为，加了self的方法表示类的方法
        :param x:输入的张量
        :param training:用来指定训练和推理的不同行为
        :return:
        '''
        batch_size = tf.shape(x)[0]
        x = self.rescale(x)
        x = self.get_grim_top_right(x)
        patches = self.extract_patches(x)
        x = self.patch_proj(patches)
        print("输入到第一个Dense，rescaling之后 x shape{}".format(x.shape))


        # broadcast将一个张量扩展成shape
        class_emb = tf.broadcast_to(input=self.class_emb, shape=[batch_size, 1, self.d_model])
        x = tf.concat([class_emb, x], axis=1)  # 沿着一个维度连接张量
        x = x + self.pos_emb

        for layer in self.enc_layers:
            x = layer(x, training)

        # First (class token) is used for classification
        #         x = self.mlp_head(x[:,0])
        print("在mlp之前，transformerblock之后shape{}".format(x.shape))
        x = self.mlp_ln(x[:, 0])
        print("取出用于分类的token后:{}".format(x.shape))
        x = self.mlp_dense1(x)
        print("在mlp_dense1后shape{}".format(x.shape))
        x = self.mlp_drop(x)
        x = self.mlp_dense2(x)
        return x


    def get_grim_top_right(self, matrix):
        '''
        求一个矩阵的右上角元素矩阵
        :param matrix: 传入一个numpy，类型为[batch,height*width,channel]
        :return: 返回右上角元素组成的numpy数组. [[1,2,3,4...]]
        '''
        batch_size = tf.shape(matrix)[0]
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

# #---------------------其他参数-------------------------
# data_path = r'/vision-transformer/Datasets/archive/fer2013.csv'
# emotions = {'0':'anger','1':'disgust','2':'fear','3':'happy','4':'sad','5':'surprised','6':'nomral'}
# logdir = 'logs'
# image_size = 48
# patch_size = 4
# num_layers = 4
# d_model = 64
# num_heads = 8
# mlp_dim = 128
# lr = 3e-4
# weight_decay = 1e-4
# batch_size = 3
# epochs = 50
# num_classes = 7
# dropout = 0.1
# image_channel = 1

# #-------------------------------train--------------------------------
# model = VisionTransformer(
#         image_size=image_size,
#         patch_size=patch_size,
#         num_layers=num_layers,
#         num_classes=num_classes,  # 类别的数量
#         d_model=d_model,  # 64
#         num_heads=num_heads,
#         mlp_dim=mlp_dim,
#         channels=image_channel,
#         dropout=dropout,
#     )

# x  = tf.random.uniform((batch_size,image_size,image_size,image_channel))
# model(x)
# print(model.inputs)
# print(model.layers)
# print(model.outputs)
# print(model.summary())


#------------------------------------数据集训练----------------------------------
#---------------------------------设置的超参数-----------------------------------
data_path = r'./Datasets/archive/fer2013.csv'
emotions = {'0':'anger','1':'disgust','2':'fear','3':'happy','4':'sad','5':'surprised','6':'nomral'}
logdir = 'logs'
image_size = 48
patch_size = 4
num_layers = 1  #TransformerBlock的数量
d_model = 64
num_heads = 8
mlp_dim = 128
lr = 3e-4
weight_decay = 1e-4
batch_size = 1
epochs = 50
num_classes = 7
dropout = 0.1
image_channel = 1
#----------------------------------设置超参数，结束-----------------------------------
AUTOTUNE = tf.data.experimental.AUTOTUNE
np.set_printoptions(precision=3,suppress=True)
# 读取csv文件
df = pd.read_csv(filepath_or_buffer=data_path, usecols=["emotion", "pixels"], dtype={"pixels": str})
fer_pixels = df.copy()

# 分成特征和标签
fer_label = fer_pixels.pop('emotion')
fer_pixels = np.asarray(fer_pixels)

# 将特征转换成模型需要的类型
fer_train = []
for i in range(len(fer_label)):
    pixels_new = np.asarray([float(p) for p in fer_pixels[i][0].split()]).reshape(48, 48, 1)
    fer_train.append(pixels_new)
fer_train = np.asarray(fer_train)
fer_label = np.asarray(fer_label)
# 转换为tf.Dateset类型
dataset = tf.data.Dataset.from_tensor_slices((fer_train, fer_label))

#数据集验证集测试集的拆分
train_dataset = dataset.take(20)
test_dataset = dataset.skip(32297)

#shuffle操作
train_dataset = (train_dataset.cache().shuffle(5 * batch_size).batch(batch_size).prefetch(AUTOTUNE))

strategy = tf.distribute.MirroredStrategy()
#模型构建
print("----------------building model---------------------------")
with strategy.scope():  # 创建一个上下文管理器，能够使用当前的训练策略
    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_classes=num_classes,  # 类别的数量
        d_model=d_model,  # 64
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        channels=image_channel,
        dropout=dropout,
    )
    # 用于配置训练方法，告知训练器使用的优化器，损失函数和准确率评测标准

    model.compile(
        # 交叉熵损失函数，from_logits=True会将结果转换成概率(softmax)，
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay),
        # 网络评价指标，如accuracy,sparse_accuracy,sparse_categorical_accuracy
        metrics=["accuracy"],)

    #最小二乘拟合
    print("-------------------start fitting-----------------------------")
    history = model.fit(
        x =train_dataset,
        epochs=50,
        callbacks=[TensorBoard(log_dir=logdir, profile_batch=0),],)