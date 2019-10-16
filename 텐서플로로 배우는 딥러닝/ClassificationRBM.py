import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets
## iris 파일 읽어들여 입력 데이터와 출력 데이터로 분리
iris = datasets.load_iris()
irisX = iris.data
irisY = iris.target
# 원-핫 벡터 생성
irisY = pd.get_dummies(irisY)
irisY = np.array(irisY)

## 이진 입력 RBM을 위한 입력 데이터의 정규화
minmax = np.amin(irisX, 0), np.amax(irisX, 0)
no_irisX = (irisX-minmax[0])/(minmax[1]-minmax[0])

## 훈련 데이터와 검정 데이터를 7:3 비율로 분리
np.random.seed(2019)
ind1 = np.random.permutation(50)
p_ind2 = np.arange(50,100)
ind2 = np.random.permutation(p_ind2)
p_ind3 = np.arange(100,150)
ind3 = np.random.permutation(p_ind3)

tr_ind1 = ind1[:35]
tr_ind2 = ind2[:35]
tr_ind3 = ind3[:35]
tr_ind = np.concatenate((tr_ind1,tr_ind2,tr_ind3),axis=0)
te_ind1 = ind1[35:]
te_ind2 = ind2[35:]
te_ind3 = ind3[35:]
te_ind = np.concatenate((te_ind1,te_ind2,te_ind3),axis=0)

trX = no_irisX[tr_ind]
teX = no_irisX[te_ind]
trY = irisY[tr_ind]
teY = irisY[te_ind]

n_input = 4
n_hidden = 20
display_step = 10
num_epochs = 200
batch_size = 5
lr = tf.constant(0.01,tf.float32)
n_class = 3

## 입력, 가중치 및 편향을 정의함
x  = tf.placeholder(tf.float32, [None, n_input], name="x")
y  = tf.placeholder(tf.float32, [None,n_class], name="y")
W_xh  = tf.Variable(tf.random_normal([n_input, n_hidden], 0.01), name="W_xh")
W_hy = tf.Variable(tf.random_normal([n_hidden,n_class], 0.01), name="W_hy")
b_i = tf.Variable(tf.zeros([1, n_input],  tf.float32, name="b_i"))
b_h = tf.Variable(tf.zeros([1, n_hidden],  tf.float32, name="b_h"))
b_y = tf.Variable(tf.zeros([1, n_class],  tf.float32, name="b_y"))


## 확률을 이산 상태, 즉 0과 1로 변환함  
def binary(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))
          
## Gibbs 표본추출 단계
def gibbs_step(x_k,y_k):
    h_k = binary(tf.sigmoid(tf.matmul(x_k, W_xh) + tf.matmul(y_k, tf.transpose(W_hy)) + b_h))
    x_k = binary(tf.sigmoid(tf.matmul(h_k, tf.transpose(W_xh)) + b_i))
    y_k = tf.nn.softmax(tf.matmul(h_k, W_hy) + b_y)
    return x_k,y_k

## 표본추출 단계 실행    
def gibbs_sample(k,x_k,y_k):
    for i in range(k):
        x_out,y_out = gibbs_step(x_k,y_k) 
## k 반복 후에 깁스 표본을 반환함
    return x_out,y_out

## CD-2 알고리즘
# 새로운 입력값
x_s,y_s = gibbs_sample(2,x,y) 
# 새로운 입력값으로 새로운 은닉노드값 구하기
act_h_s = tf.sigmoid(tf.matmul(x_s,W_xh) + tf.matmul(y_s,tf.transpose(W_hy)) + b_h) 
# 입력값으로 은닉노드값 구하기
act_h = tf.sigmoid(tf.matmul(x,W_xh) + tf.matmul(y,tf.transpose(W_hy)) + b_h) 
# 은닉노드값이 주어질때 입력값을 추출
_x = binary(tf.sigmoid(tf.matmul(act_h,tf.transpose(W_xh)) + b_i)) 


# 가중치와 편향을 업데이트 한다.
W_xh_add = tf.multiply(lr/batch_size,tf.subtract(tf.matmul(tf.transpose(x),act_h), tf.matmul(tf.transpose(x_s),act_h_s)))
W_hy_add = tf.multiply(lr/batch_size,tf.subtract(tf.matmul(tf.transpose(act_h),y), tf.matmul(tf.transpose(act_h_s),y_s)))
bi_add = tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(x,x_s),0,True))
bh_add = tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(act_h,act_h_s),0,True))
by_add = tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(y,y_s),0,True))

updt = [W_xh.assign_add(W_xh_add),W_hy.assign_add(W_hy_add),b_i.assign_add(bi_add),b_h.assign_add(bh_add),b_y.assign_add(by_add)]

# 텐서플로우 그래프를 실행시킨다.
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(num_epochs):
        ind2 = np.random.permutation(len(trX))
        num_batch = int(len(trX)/batch_size)

        for i in range(num_batch):
            batch_xs = trX[ind2[i*batch_size:(i+1)*batch_size]]
            batch_ys = trY[ind2[i*batch_size:(i+1)*batch_size]]

            _ = sess.run([updt],feed_dict={x:batch_xs,y:batch_ys})

        if epoch % display_step == 0:
            print("Epoch:",'%04d'%(epoch+10))

    print("Discriminative RBM training Completed !")
    
    # 훈련데이터 정확도 계산
    tr_lab1 = np.zeros((len(trX),n_class)); tr_lab1[:,0] = 1
    tr_lab2 = np.zeros((len(trX),n_class)); tr_lab2[:,1] = 1
    tr_lab3 = np.zeros((len(trX),n_class)); tr_lab3[:,2] = 1

    tr_f1_x1 = tf.reduce_sum(tf.nn.softplus(tf.matmul(tf.cast(trX,tf.float32),W_xh) + tf.matmul(tf.cast(tr_lab1,tf.float32),tf.transpose(W_hy)) + b_h),1)
    tr_f2_x1 = tf.reduce_sum(tf.nn.softplus(tf.matmul(tf.cast(trX,tf.float32),W_xh) + tf.matmul(tf.cast(tr_lab2,tf.float32),tf.transpose(W_hy)) + b_h),1)
    tr_f3_x1 = tf.reduce_sum(tf.nn.softplus(tf.matmul(tf.cast(trX,tf.float32),W_xh) + tf.matmul(tf.cast(tr_lab3,tf.float32),tf.transpose(W_hy)) + b_h),1)

    tr_f_x1 = b_y + tf.transpose([tr_f1_x1,
                                  tr_f2_x1,
                                  tr_f3_x1])

    tr_y_hat = tf.nn.softmax(tr_f_x1)

    tr_correct_pred = tf.equal(tf.argmax(tr_y_hat,1),tf.argmax(trY,1))
    tr_accuracy = tf.reduce_mean(tf.cast(tr_correct_pred,tf.float32))

    print("Training Accuracy:",sess.run(tr_accuracy))

    # 검증데이터 정확도 계산
    te_lab1 = np.zeros((len(teX),n_class)); te_lab1[:,0] = 1
    te_lab2 = np.zeros((len(teX),n_class)); te_lab2[:,1] = 1
    te_lab3 = np.zeros((len(teX),n_class)); te_lab3[:,2] = 1

    te_f1_x1 = tf.reduce_sum(tf.nn.softplus(tf.matmul(tf.cast(teX,tf.float32),W_xh) + tf.matmul(tf.cast(te_lab1,tf.float32),tf.transpose(W_hy)) + b_h),1)
    te_f2_x1 = tf.reduce_sum(tf.nn.softplus(tf.matmul(tf.cast(teX,tf.float32),W_xh) + tf.matmul(tf.cast(te_lab2,tf.float32),tf.transpose(W_hy)) + b_h),1)
    te_f3_x1 = tf.reduce_sum(tf.nn.softplus(tf.matmul(tf.cast(teX,tf.float32),W_xh) + tf.matmul(tf.cast(te_lab3,tf.float32),tf.transpose(W_hy)) + b_h),1)

    te_f_x1 = b_y + tf.transpose([te_f1_x1,
                                  te_f2_x1,
                                  te_f3_x1])

    te_y_hat = tf.nn.softmax(te_f_x1)

    te_correct_pred = tf.equal(tf.argmax(te_y_hat,1),tf.argmax(teY,1))
    te_accuracy = tf.reduce_mean(tf.cast(te_correct_pred,tf.float32))

    print("Testing Accuracy:",sess.run(te_accuracy))

