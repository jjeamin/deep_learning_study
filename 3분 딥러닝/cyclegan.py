import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


os.chdir('C:/Users/woals/Downloads/vangogh2photo/vangogh2photo')
img = cv2.imread('trainA/00001.jpg',cv2.IMREAD_COLOR)
print(img.shape)

cv2.imshow('test',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

epoch = 200
epoch_step = 100
batch_size = 1
learning_rate = 0.0001

real = tf.placeholder(tf.float32,shape=[None,256,256,6])
fake = tf.placeholder(tf.float32,shpae=[None,256,256,3])

