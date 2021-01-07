# -*- coding: utf-8 -*-

'''
Created on 2021/1/7  17:00

@project: OCR

@filename: LeNet_test.py

@author: knavezl

@Desc:    
    LeNet-5模型 :
    https://blog.csdn.net/weixin_41695564/article/details/80240106
    https://my.oschina.net/u/876354/blog/1926060
    https://my.oschina.net/u/876354/blog/1632862
'''
 
import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
 
###加载LeNet.py和mnist_train.py中定义的常量和前向传播的函数########
import LeNet
import LeNet_train
 
#测试
def test( mnist ):
    with tf.Graph().as_default() as g:   #将默认图设为g
        xs = mnist.test.images
        ys = mnist.test.labels
        print('xs.shape[0]=', xs.shape[0])  # 10000

        #定义输入的格式
        x = tf.placeholder(tf.float32, [1,#此处设置为1是单张图片进行测试，可根据自己需要进行批量读取
                                        LeNet.IMAGE_SIZE,
                                        LeNet.IMAGE_SIZE,
                                        LeNet.NUM_CHANNELS], name='x-input1')

        #直接通过调用封装好的函数来计算前向传播的结果
        #测试时不关注过拟合问题，所以正则化输入为None
        logit = LeNet.Model(x,None, None)
 
        #通过变量重命名的方式来加载模型
        variable_averages = tf.train.ExponentialMovingAverage(LeNet_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        # 所有滑动平均的值组成的字典，处在/ExponentialMovingAverage下的值
        # 为了方便加载时重命名滑动平均量，tf.train.ExponentialMovingAverage类
        # 提供了variables_to_store函数来生成tf.train.Saver类所需要的变量
        saver = tf.train.Saver(variable_to_restore) #这些值要从模型中提取
        
        
        # 保留概率，用于 dropout 层
        keep_prob = tf.placeholder(tf.float32)
 
        with tf.Session() as sess:
            #tf.train.get_checkpoint_state函数
            # 会通过checkpoint文件自动找到目录中最新模型的文件名
            model_dir=r"models/mnist_model"
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                #加载模型
                #saver.restore(sess, ckpt.model_checkpoint_path)
                saver.restore(sess, tf.train.latest_checkpoint(model_dir))
                for test_idx in range(len(xs)):
                    # 类似地将输入的测试数据格式调整为一个四维矩阵
                    x_image = np.reshape(xs[test_idx], (LeNet.IMAGE_SIZE,
                                                  LeNet.IMAGE_SIZE,
                                                  LeNet.NUM_CHANNELS))

                    y_label=np.argmax(ys[test_idx])

                    # 跑模型进行识别
                    y_conv = tf.nn.softmax(logit)
                    y_conv = tf.argmax(y_conv,1)
                    pred=sess.run(y_conv,feed_dict={x:[x_image], keep_prob: 1.0})
                    if y_label==pred[0]:
                        pass
                    else:
                        print('测试第',test_idx+1,'条数据，正确：',y_label,'，预测：',pred[0])
            else:
                print("No checkpoint file found")

def main( argv=None ):
    mnist = input_data.read_data_sets("data/mnist/", one_hot=True)
    test(mnist)
 
if __name__=='__main__':
    tf.app.run()