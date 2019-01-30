import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data",one_hot=True)

img = mnist.test.images[1]

print(img.shape)

'''
learning_rate = 0.01
training_epoch = 20
batch_size = 100
n_hidden = 256

X = tf.placeholder(tf.float32,shape=[None,28*28]) #input

W_encode = tf.Variable(tf.truncated_normal([28*28,256]))
b_encode = tf.Variable(tf.truncated_normal([256]))

encoder = tf.nn.sigmoid(tf.add(tf.matmul(X,W_encode),b_encode))

W_decode = tf.Variable(tf.truncated_normal([256,28*28]))
b_decode = tf.Variable(tf.truncated_normal([28*28]))

decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder,W_decode),b_decode))

cost = tf.reduce_mean(tf.pow(X - decoder,2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    total_batch = int(mnist.train.num_examples / batch_size)
    print(total_batch)

    for epoch in range(training_epoch):
        total_cost = 0

        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            _,cost_val = sess.run([optimizer,cost],feed_dict={X:batch_xs})
            total_cost += cost_val

        print('Epoch','%04d' % (epoch + 1),'Avg, cost =','{:.4f}'.format(total_cost / total_batch))

    print('success')


    sample_size = 10

    samples = sess.run(decoder,feed_dict={X:mnist.test.images[:sample_size]})

    fig, ax = plt.subplots(2,sample_size,figsize=(sample_size,2))

    

    for i in range(sample_size):
        ax[0][i].set_axis_off()
        ax[1][i].set_axis_off()
        ax[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        ax[1][i].imshow(np.reshape(samples[i],(28,28)))

    plt.show()
'''