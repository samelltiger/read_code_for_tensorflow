import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt


def weight_init(shape):
    initial = tf.random_uniform(shape=shape, dtype=tf.float32, minval=-np.sqrt(5) / np.sqrt(shape[0]),
                                maxval=np.sqrt(5) / np.sqrt(shape[0]))
    return tf.Variable(initial_value=initial, dtype=tf.float32)


# 全部初始化为 0
def zero_init(shape):
    initial = tf.Variable(tf.zeros(shape))
    return tf.Variable(initial, trainable=True)


# 正交矩阵初始化
def orthogonal_initializer(shape, scale=1.0):
    scale = 1.0
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)  # this needs to be corrected to float32
    return tf.Variable(scale * q[:shape[0], :shape[1]], trainable=True, dtype=tf.float32)


def bias_init(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


# 洗牌
def shufflelists(data):
    ri = np.random.permutation(len(data))
    data = [data[i] for i in ri]
    return data


def Standardize(seq):
    centrized = seq - np.mean(seq, axis=0)
    normalizerd = centrized / np.std(seq, axis=0)
    return normalizerd


class LSTM(object):
    def __init__(self, incoming, D_input, D_cell, initializer, f_bias=1.0):
        """

        :param incoming: 用来接收输入数据的，其形状为 [n_samples,n_steps,D_cell ]
        :param D_input:  输入的维度
        :param D_cell:   LSTM的hidden state的维度，同时也是memory cell的维度
        :param initializer:
        :param f_bias:
        """
        self.incoming = incoming
        self.D_input = D_input
        self.D_cell = D_cell
        # 输入门的三个参数
        # igate = w_xi.*x + W_hi. * h + b_i
        self.W_xi = initializer([self.D_input, self.D_cell])
        self.W_hi = initializer([self.D_cell, self.D_cell])
        self.b_i = tf.Variable(tf.zeros([self.D_cell]))

        # 遗忘门的三个参数
        # fgate = W_xf. * x + W_hf. * h +b_f
        self.W_xf = initializer([self.D_input, self.D_cell])
        self.W_hf = initializer([self.D_cell, self.D_cell])
        self.b_f = tf.Variable(tf.constant(f_bias, shape=[self.D_cell]))

        # 输出门参数
        self.W_xo = initializer([self.D_input, self.D_cell])
        self.W_ho = initializer([self.D_cell, self.D_cell])
        self.b_o = tf.Variable(tf.zeros(self.D_cell))

        # 计算新信息的三个参数
        # cell = W_xc.*x+W_hc.*h +b_c
        self.W_xc = initializer([self.D_input, self.D_cell])
        self.W_hc = initializer([self.D_cell, self.D_cell])
        self.b_c = tf.Variable(tf.zeros([self.D_cell]))

        # 最初时的hidden state 和 memory cell 的值，二者的形状都是 [n_samples,D_cell]
        # 如果没有特殊指定，这里直接设成全部为0
        init_for_both = tf.matmul(self.incoming[:, 0, :], tf.zeros([self.D_input, self.D_cell]))
        self.hid_init = init_for_both
        self.cell_init = init_for_both
        # 所以要将hidden state 和 memory 并在一起
        self.previous_h_c_tuple = tf.stack([self.hid_init, self.cell_init])
        # 需要将数据由 [ n_samples,n_steps,D_cell] 的形状变成 [n_steps,n_sameples,D_cell] 的形状
        self.incoming = tf.transpose(self.incoming, perm=[1, 0, 2])

    def one_step(self, previous_h_c_tuple, current_x):
        # 在将hidden state 和 memory cell 拆分开
        prev_h, prev_c = tf.unstack(previous_h_c_tuple)
        # 这时，current时当前的输入
        # prev_h 是上一个时刻的 hidden state
        # prev_c 是上一个时刻的 memory cell

        # 计算输入门
        i = tf.sigmoid(
            tf.matmul(current_x, self.W_xi) + tf.matmul(prev_h, self.W_hi) + self.b_i
        )
        # 计算遗忘门
        f = tf.sigmoid(
            tf.matmul(current_x, self.W_xf) + tf.matmul(prev_h, self.W_hf) + self.b_f
        )
        # 计算输出门
        o = tf.sigmoid(
            tf.matmul(current_x, self.W_xo) + tf.matmul(prev_h, self.W_ho) + self.b_o
        )
        # 计算新的数据来源
        c = tf.tanh(
            tf.matmul(current_x, self.W_xc) + tf.matmul(prev_h, self.W_hc) + self.b_c
        )
        # 计算当前时刻的 memory cell
        current_c = f * prev_c + i * c
        # 计算当前时刻的 hidden state
        current_h = o * tf.tanh(current_c)
        # 再次将当前的hidden state和memory cell 并在一起返回
        return tf.stack([current_h, current_c])

    def all_steps(self):
        # 输出形状：[n_steps,n_sample,D_cell]
        hstates = tf.scan(fn=self.one_step,
                          elems=self.incoming,  # 形状为[n_steps,n_sample,D_input]
                          initializer=self.previous_h_c_tuple,
                          name="hstates"
                          )[:, 0, :, :]
        return hstates


# 读取输入和输出数据
mfc = np.load("X.npy", encoding="bytes")
art = np.load("Y.npy", encoding="bytes")
totalsamples = len(mfc)
print("totalsamples:",totalsamples)
# 20% 的数据作为 validation set
vali_size = 0.2


# 将每个样本的输入和输出数据合成list，在将所有的样本合成list
# 其中输入数据的形状是：[n_samples,n_steps,D_input]
# 其中输出数据的形状是：[n_samples,D_output]
def data_prer(X, Y):
    D_input = X[0].shape[1]
    data = []
    for x, y in zip(X, Y):
        data.append([Standardize(x).reshape((1, -1, D_input)).astype("float32"),
                     Standardize(y).astype("float32")])

    return data


# 处理数据
data = data_prer(mfc, art)
# 分训练集和验证集
train = data[int(totalsamples * vali_size):]
test = data[:int(totalsamples * vali_size)]
print("num of train squences:%s"%len(train))
print("num of test sequences:%s"%len(test))
print("shape of inputs:",test[0][0].shape)
print("shape of labels:",test[0][1].shape)

# 构造网络
D_input = 39
D_label = 24
learning_rate = 7e-5
num_units = 1024
# 样本的输入和标签
inputs = tf.placeholder(tf.float32, [None, None, D_input], name="inputs")
labels = tf.placeholder(tf.float32, [None, D_label], name="labels")

# 实例化LSTM类
rnn_cell = LSTM(inputs, D_input, num_units, orthogonal_initializer)
# 调用scan计算所有hidden states
rnn0 = rnn_cell.all_steps()
# 将3维tensor [n_steps,n_samples,D_cell]转成 矩阵 [n_steps*n_samples,D_cell]
# 用于计算outputs
rnn = tf.reshape(rnn0, [-1, num_units])
# 输出层的学习参数
W = weight_init([num_units, D_label])
b = bias_init([D_label])
output = tf.matmul(rnn, W) + b
# 损失
loss = tf.reduce_mean((output - labels) ** 2)
# 训练所需
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 训练网络
# 建立session并实际初始化所有参数
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# 训练并记录
def train_epoch(EPOCH):
    for k in range(EPOCH):
        train0 = shufflelists(train)
        if k == 0:
            print("train_shape:", len(train))
            print("train0_shape:", len(train0))
            print("train0_shape:", len(train[0][0]))
            print("train0_label_shape:", len(train0[0][1]))
        for i in range(len(train)):
            sess.run(train_step, feed_dict={inputs: train0[i][0], labels: train0[i][1]})
        tl = 0
        dl = 0
        for i in range(len(test)):
            dl += sess.run(loss, feed_dict={inputs: test[i][0], labels: test[i][1]})

        for i in range(len(train)):
            tl += sess.run(loss, feed_dict={inputs: train[i][0], labels: train[i][1]})

        print(k, "train:", round(tl / 83, 3), "test:", round(dl / 20, 3))


t0 = time.time()
train_epoch(10)
t1 = time.time()
print(" %f seconds:" % round(t1 - t0, 2))


pY = sess.run(output,feed_dict={inputs:test[10][0]})
plt.plot(pY[:,8])
plt.plot(test[10][1][:,8])
plt.title("test")
plt.legend(['predicted','real'])


sess.close()