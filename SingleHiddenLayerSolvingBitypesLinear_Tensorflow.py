
# coding: utf-8

# In[20]:

#单隐层网络解决二分类问题#

import tensorflow as tf
from numpy.random import RandomState

#定义训练数据batch大小
batch_size = 8

#定义神经网络参数,均值为0，标准差为1，2*3矩阵与均值为0，标准差为1，3*1矩阵
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#在shape一个维度直接使用none可以方便使用不大的batch大小，但数据集较大可能会内存溢出
x = tf.placeholder(tf.float32, shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None,1),name='y-input')

#定义前向传播过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#定义损失函数和反向传播
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#通过随机数生成模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)

#定义规则给出样本标签，这里所有x1+x2<1都认为是正样本
Y = [[int(x1+x2<1)]for (x1,x2) in X]

#创建一个会话来运行TF程序
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print("w1=%a"%(sess.run(w1)))
    print("w2=%a"%(sess.run(w2)))

    #设定训练轮数
    STEPS=10000
    for i in range(STEPS):
        #每次选择batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size , dataset_size)
        
        #通过选取的样本训练NN并更新参数
        sess.run(train_step, feed_dict={x:X[start:end],y_:Y[start:end]})
        if i% 1000 == 0:
        #每隔一段时间计算交叉熵输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d trainging step(s), cross_entropy on all data is %g"%(i, total_cross_entropy))
    print("w1=%a"%(sess.run(w1)))
    print("w2=%a"%(sess.run(w2)))
        


# In[ ]:



