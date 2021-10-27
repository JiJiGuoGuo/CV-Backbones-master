#此程序用来配置GhostEfficientNetV2的
import tensorflow as tf
import EfficientNetV2 as effi
import numpy as np
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import pandas as pd
#---------------------------------设置的超参数-----------------------------------
data_path = r'../../Datasets/archive/fer2013.csv'
emotions = {'0':'anger','1':'disgust','2':'fear','3':'happy','4':'sad','5':'surprised','6':'nomral'}
logdir = 'logs'
image_size = 48
patch_size = 4
num_layers = 4
d_model = 64
num_heads = 8
mlp_dim = 128
lr = 3e-4
weight_decay = 1e-4
batch_size = 3 #batch设置2的幂次对GPU表现更优，batch_size通常在几十到几百
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
    pixels_new = np.asarray([float(p) for p in fer_pixels[i][0].split()]).reshape(48,48,1)
    fer_train.append(pixels_new)
fer_train = np.asarray(fer_train)
fer_label = np.asarray(fer_label)
# 转换为tf.Dateset类型
dataset = tf.data.Dataset.from_tensor_slices((fer_train, fer_label))

#数据集验证集测试集的拆分
train_dataset = dataset.take(200)
test_dataset = dataset.skip(32297)

#shuffle操作
train_dataset = (train_dataset.cache().shuffle(5 * batch_size).batch(batch_size).prefetch(AUTOTUNE))

strategy = tf.distribute.MirroredStrategy()
#模型构建
print("----------------building model---------------------------")
with strategy.scope():  # 创建一个上下文管理器，能够使用当前的训练策略
    model = effi.EfficientNetV2_S(num_classes=7,activation='swish',width_mult=1.0,depth_mult=1.0,
                                conv_dropout_rate=None,dropout_rate=None,drop_connect=0.2)
    # 用于配置训练方法，告知训练器使用的优化器，损失函数和准确率评测标准
    model.compile(
        # 交叉熵损失函数，from_logits=True会将结果转换成概率(softmax)，
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay),
        # 网络评价指标，如accuracy,sparse_accuracy,sparse_categorical_accuracy
        metrics=["accuracy"],)


    # 模型断点续训
    checkpoint_filepath = '/tmp/checkpoint' #断点文件保存路径
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                  save_weights_only=True,
                                                  monitor='val_accuracy',
                                                  model='max',
                                                  save_best_only=True)
    #最小二乘拟合
    print("-------------------start fitting-----------------------------")
    history = model.fit(x=train_dataset,shuffle=True,epochs=50,callback=[model_checkpoint_callback])
    model.load_weights(checkpoint_filepath)

    model.summary()

plt.subplot(2,1,1)
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')

plt.show()