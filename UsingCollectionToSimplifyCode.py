
# coding: utf-8

# In[2]:

import tensorflow as tf
#获取一层神经网络边上的权重，并将这个权重的L2正则化加入“losses”的集合中
def get_weight(shape, lambda):
    #生成一个变量
    var = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
    
    #add_to_collection函数将这个新生成变量的L2正则化损失项加入集合。
    #这个函数的第一个参数‘losses’是集合的名字，第二个参数是要加入集合的内容
    tf.add_to_collection('losses',tf.contrib.layers.12_regularizer(lambda)(var))
    
    #返回生成的变量
    return var

x = tf.placeholder(tf.float32, shape = (None, 2))
y_ = tf.placeholder(tf.float32, shape = (None, 1))
batch_size = 8

#定义了每一层网络中节点的个数
layer_dimension = [2, 10, 10, 10, 1]
#神经网络的层数
n_layers = len(layer_dimension)

#这个变量维护前向传播最深层的节点，开始就是输入层
cur_layer = x
#当前层的节点个数
in_dimension = layer_dimension[0]

#生成五层全连接神经网络
for i in range(1, n_layers):
    out_dimension = layer_dimension[i]
    #生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图上的集合
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    #使用RELU为激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    #进入下一层之前将下一层的节点个数更新为当前层节点个数
    in_dimension = layer_dimension[i]

mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

#将均方误差损失加入损失集合
tf.add_to_collection('losses', mse_loss)

#get_collection返回一个列表，这个列表是所集合上的元素。在这段代码中，元素是损失函数的不同部分，相加得最终损失函数
loss = tf.add_n(tf.get_collection('losses'))


# In[ ]:



