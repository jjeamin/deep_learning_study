import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data",one_hot=True)

total_epoch = 100
batch_size = 100
learning_rate = 0.0001
n_hidden = 256
n_input = 28*28
n_noise = 128

X = tf.placeholder(tf.float32,[None,n_input])
Z = tf.placeholder(tf.float32,[None,n_noise])

#Generate
G_W1 = tf.Variable(tf.truncated_normal([n_noise,n_hidden],stddev=0.01))
G_b1 = tf.Variable(tf.zeros(n_hidden))
G_W2 = tf.Variable(tf.truncated_normal([n_hidden,n_input],stddev=0.01))
G_b2 = tf.Variable(tf.zeros(n_input))

#Discriminator
D_W1 = tf.Variable(tf.truncated_normal([n_input,n_hidden],stddev=0.01))
D_b1 = tf.Variable(tf.zeros(n_hidden))
D_W2 = tf.Variable(tf.truncated_normal([n_hidden,1],stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))

#가짜 이미지 만들기
def generator(noise):
    hidden = tf.nn.relu(tf.matmul(noise,G_W1)+G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden,G_W2)+G_b2)
    return output

#진짜 이미지
def discriminator(input):
    hidden = tf.nn.relu(tf.matmul(input,D_W1)+D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden,D_W2)+D_b2)
    return output

def get_noise(batch_size,n_noise):
    return np.random.normal(size=(batch_size,n_noise))

G = generator(Z) #가짜 이미지 생성
D_gene = discriminator(G) # 가짜 이미지 학습
D_real = discriminator(X) # 진짜 이미지 학습

loss_D = tf.reduce_mean(tf.log(D_real)+tf.log(1-D_gene))
loss_G = tf.reduce_mean(D_gene)

D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

# GAN 논문의 수식에 따르면 loss 를 극대화 해야하지만, minimize 하는 최적화 함수를 사용하기 때문에
# 최적화 하려는 loss_D 와 loss_G 에 음수 부호를 붙여줍니다.
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D,
                                                         var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G,
                                                         var_list=G_var_list)

#########
# 신경망 모델 학습
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        # 판별기와 생성기 신경망을 각각 학습시킵니다.
        _, loss_val_D = sess.run([train_D, loss_D],
                                 feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G],
                                 feed_dict={Z: noise})

    print('Epoch:', '%04d' % epoch,
          'D loss: {:.4}'.format(loss_val_D),
          'G loss: {:.4}'.format(loss_val_G))

    #########
    # 학습이 되어가는 모습을 보기 위해 주기적으로 이미지를 생성하여 저장
    ######
    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, feed_dict={Z: noise})

        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('최적화 완료!')
