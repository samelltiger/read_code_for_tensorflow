'''
本次的代码LV2是紧接着代码LV1的升级版，所学习的内容与先前的一样，不同的是：

简单梳理调整了代码结构，方便使用
将所有gate的计算并在一个大矩阵乘法下完成提高GPU的利用率
除了LSTM（Long-Short Term Memory）以外的cell，还提供了GRU（gate recurrent unit） cell模块
双向RNN（可选择任意cell组合）
该代码可被用于练习结构改造或实际建模任务
'''

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt


class LSTMcell(object):
    def __init__(self, incoming, D_input, D_cell, initializer, f_bias=1.0, L2=False, h_act=tf.tanh, init_h=None,
                 init_c=None):
        self.incoming = incoming
        self.D_input = D_input
        self.D_cell = D_cell
        self.initializer = initializer  # 初始化方法
        self.f_bias = f_bias  # 遗忘门的初始偏移量
        self.h_act = h_act  # 这里可以选择LSTM的hidden state的激活函数
        self.type = 'lstm'  # 区分gru
        # 如果没有提供最初的 hidden state 和 memory cell ，会全部初始化为 0
        if init_h is None and init_c is None:
            self.init_h = tf.matmul(self.incoming[0, :, :], tf.zeros([self.D_input, self.D_cell]))
            self.init_c = self.init_h
            self.previous = tf.stack([self.init_h, self.init_c])

        # LSTM 所有需要学习的参数，每个都是[W_x,W_h,b_f]的tuple
        self.igate = self.Gate()
        self.fgate = self.Gate(bias=f_bias)
        self.ogate = self.Gate()
        self.cell = self.Gate()
        # 因为所有的gate都会乘以当前的输出和上一时刻的hidden state
        # 将矩阵concat在一起，计算后注意分离，加快运行速度
        # W_x 的形状是 [D_input,4*D_cell]
        self.W_x = tf.concat(values=[self.igate[0], self.fgate[0], self.ogate[0], self.cell[0]], axis=1)
        self.W_h = tf.concat(values=[self.igate[1], self.fgate[1], self.ogate[1], self.cell[1]], axis=1)
        self.b = tf.concat(values=[self.igate[2], self.fgate[2], self.ogate[2], self.cell[2]], axis=0)
        # 对LSTM的权重进行 L2 regularization
        if L2:
            self.L2_loss = tf.nn.l2_loss(self.W_x) + tf.nn.l2_loss(self.W_h)

    # 初始化gate函数
    def Gate(self, bias=0.001):
        Wx = self.initializer([self.D_input, self.D_cell])
        Wh = self.initializer([self.D_cell, self.D_cell])
        b = tf.Variable(tf.constant(bias, shape=[self.D_cell]), trainable=True)
        return Wx, Wh, b

    # 大矩阵乘法运算完毕后，方便用于分离各个gate
    def Slice(self, x, n):
        return x[:, n * self.D_cell:(n + 1) * self.D_cell]

    # 每个time step 需要运行的步骤
    def Step(self, previous_h_c_tuple, current_x):
        # 分离上一时刻的 hidden state 和 memory cell
        prev_h, prev_c = tf.unstack(previous_h_c_tuple)
        # 统一在concat 成的大矩阵中一次完成所有的gates计算
        gates = tf.matmul(current_x, self.W_x) + tf.matmul(prev_h, self.W_h) + self.b
        # 分离输入门
        i = tf.sigmoid(self.Slice(gates, 0))
        # 分离遗忘门
        f = tf.sigmoid(self.Slice(gates, 1))
        # 分离输出门
        o = tf.sigmoid(self.Slice(gates, 2))
        # 分离新的更新信息
        c = tf.tanh(self.Slice(gates, 3))
        # 利用gates进行当前memory cell的计算
        current_c = f * prev_c + i * c
        # 利用gates进行当前hidden state的计算
        current_h = o * self.h_act(current_c)
        return tf.stack(current_h, current_c)


# 定义GRUcell类
class GRUcell(object):
    def __init__(self, incoming, D_input, D_cell, initializer, L2=False, init_h=None):
        self.incoming = incoming
        self.D_input = D_input
        self.D_cell = D_cell
        self.initializer = initializer
        self.type = "gru"
        # 如果没有提供最初的 hidden state 会初始化为 0
        # 注意GRU中并没有LSTM 中的 memory cell ，其功能是由 hidden state 完成的
        if init_h is None:
            # If init_h is not provided,initialize it
            # the shape of init_h is [n_samples,D_cell]
            self.init_h = tf.matmul(self.incoming[0, :, :], tf.zeros([self.D_input, self.D_cell]))
            self.previous = self.init_h

        # 如果没有提供最初的hidden state ，会初始化为0
        # 注意，GRU中并没有LSTM中的memory cell ，其功能是有 hidden state完成的
        self.rgate = self.Gate()
        self.ugate = self.Gate()
        self.cell = self.Gate()

        # 因为所有的gate都会乘以当前的输入和上一时刻的 hidden state
        # 将矩阵concat在一起，计算后在逐一分离，加快运行速度
        # W_x 的形状是[D_input,3*D_cell]
        self.W_x = tf.concat(values=[self.rgate[0], self.ugate[0], self.cell[0]], axis=1)
        self.W_h = tf.concat(values=[self.rgate[1], self.ugate[1], self.cell[1]], axis=1)
        self.b = tf.concat(values=[self.rgate[2], self.ugate[2], self.cell[2]], axis=0)
        # 对LSTM的权重进行L2 regularization
        if L2:
            self.L2_loss = tf.nn.l2_loss(self.W_x) + tf.nn.l2_loss(self.W_h)

    # 初始化gate函数
    def Gate(self, bias=0.001):
        Wx = self.initializer([self.D_input, self.D_cell])
        Wh = self.initializer([self.D_cell, self.D_cell])
        b = tf.Variable(tf.constant(bias, shape=[self.D_cell]), trainable=True)
        return Wx, Wh, b

    # 大矩阵乘法运算完毕后，方便用于分离各个gate
    def Slice_W(self, x, n):
        return x[:, n * self.D_cell :(n+1) * self.D_cell]

    # 每个time step 需要运行的步骤
    def Step(self, prev_h, current_x):
        # 分两次 ，统一在concat成的大矩阵中完成的gates所需要的计算
        Wx = tf.matmul(current_x, self.W_x) + self.b
        Wh = tf.matmul(prev_h, self.W_h)
        # 分离和组合reset gate
        r = tf.sigmoid(self.Slice_W(Wx, 0)+ self.Slice_W(Wh, 0))
        # 分离组合 update gate
        u = tf.sigmoid(self.Slice_W(Wx, 1) + self.Slice_W(Wh, 1))
        # 分离和组合新的更新信息
        # 注意GRU中，在这一步就已经有 reset gate 的干涉了
        c = tf.tanh(self.Slice_W(Wx, 2) + r * self.Slice_W(Wh, 2))
        # 计算当前hidden state ，GRU 将LSTM 中的 input gate 和output gate的和设成 1
        # 用 update gate 完成两者的工作
        current_h = (1 - u) * prev_h + u * c
        return current_h


# 定义RNN函数
def RNN(cell, cell_b=None, merge="sum"):
    '''
    该函数接收的数据需要是 [n_steps, n_sample,D_output]
    函数的输入也是 [n_steps,n_sample,D_output]
    如果输入数据不是[ n_steps,n_sample,D_input]
    使用 inputs_T = tf.transpose(inputs,perm[1,0,2])
    :param cell:
    :param cell_b:
    :param merge:
    :return:
    '''
    hstates = tf.scan(fn=cell.Step,
                      elems=cell.incoming,
                      initializer=cell.previous,
                      name='hstates')
    '''
    lstm 的step 经过scan 计算后会返回4维tensor，
    其中[:,0,:,:]表示hidden state
    [:,1,:,:]表示memory cell，这里只需要hidden state
    '''
    if cell.type == "lstm":
        hstates = hstates[:, 0, :, :]
    # 如果提供了第二个cell，将进行反向rnn的计算
    if cell_b is not None:
        # 将数据变为反向
        incoming_b = tf.reverse(cell.incoming, axis=[0])
        # scan 计算反向rnn
        b_hstates_rev = tf.scan(fn=cell_b.Step,
                                elems=incoming_b,
                                initializer=cell_b.previous,  # 每个cell自带的初始值
                                name="b_hstates"
                                )
        if cell_b.type == "lstm":
            b_hstates_rev = b_hstates_rev[:, 0, :, :]
        # 用scan 计算好的反向rnn需要再反向回来与正向rnn所计算的数据进行合并
        b_hstates = tf.reverse(b_hstates_rev, axis=[0])
        # 合并方式可以选择直接相加，也可以选择concat
        if merge == "sum":
            hstates = hstates + b_hstates
        else:
            hstates = tf.concat(values=[hstates, b_hstates], axis=2)
    return hstates


def weight_init(shape):
    initial = tf.random_uniform(shape, minval=-np.sqrt(5) * np.sqrt(1.0 / shape[0]),
                                maxval=np.sqrt(5) * np.sqrt(1.0 / shape[0]))
    return tf.Variable(initial, trainable=True)


def zero_init(shape):
    initial = tf.Variable(tf.zeros(shape))
    return tf.Variable(initial, trainable=True)


def orthogonal_initializer(shape, scale=1.0):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return tf.Variable(scale * q[:shape[0], :shape[1]], dtype=tf.float32, trainable=True)


def bias_init(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, trainable=True)


def shufflelists(data):
    ri = np.random.permutation(len(data))
    data = [data[i] for i in ri]
    return data


def Standardize(seq):
    centerized = seq - np.mean(seq, axis=0)
    normalized = centerized / np.std(centerized, axis=0)
    return normalized


mfc = np.load("X.npy", encoding='bytes')
art = np.load("Y.npy", encoding='bytes')
print("mfc.shape:",mfc.shape)
print("art.shape:",art.shape)
print("mfc[0].shape:",mfc[0].shape)
print("art[0].shape:",art[0].shape)
# print("art.shape:",art.shape)
# print("art.shape:",art.shape)
totalsamples = len(mfc)
vali_size = 0.2


def data_prer(X, Y):
    D_input = X[0].shape[1]
    D_output = 24
    data = []
    for x, y in zip(X, Y):
        data.append([Standardize(x).reshape((1, -1, D_input)).astype("float32"),
                     Standardize(y).astype('float32')])
    return data


data = data_prer(mfc, art)
train = data[int(totalsamples * vali_size):]
test = data[:int(totalsamples * vali_size)]

print("num of train sequences: %s" % len(train))
print("num of test sequences: %s" % len(test))
print("shape of inputs(test[0][0].shape):", test[0][0].shape)
print("shape of labels(test[0][1].shape):", test[0][1].shape)

# 执行代码
D_input = 39
D_label = 24
learning_rate = 7e-5
num_units = 1024
L2_penalty = 1e-4
inputs = tf.placeholder(tf.float32, [None, None, D_input], name="inputs")
labels = tf.placeholder(tf.float32, [None, D_label], name="labels")
# 保持多少节点不被dropout掉
drop_keep_rate = tf.placeholder(tf.float32, name="dropout_keep")
# 用于 reshape
n_steps = tf.shape(inputs)[1]
n_samples = tf.shape(inputs)[0]
# 将输入数据从 [n_samples,n_steps,D_input],reshape 成 [n_samples*n_steps,D_input]
# 用于feedforward layer 的使用
re1 = tf.reshape(inputs, [-1, D_input])
# 第一层
Wf0 = weight_init([D_input, num_units])
bf0 = bias_init([num_units])
h1 = tf.nn.relu(tf.matmul(re1, Wf0) + bf0)
# dropout
h1d = tf.nn.dropout(h1, drop_keep_rate)
# 第二层
Wf1 = weight_init([num_units, num_units])
bf1 = bias_init([num_units])
h2 = tf.nn.relu(tf.matmul(h1d, Wf1) + bf1)
# dropout
h2d = tf.nn.dropout(h2, drop_keep_rate)
# 将输入数据从 [n_samples*n_steps,D_input],reshape成[n_samples,n_steps,D_input]
# 用于双向 rnn layer 的使用
re2 = tf.reshape(h2d, [n_samples, n_steps, num_units])
# 将数据从 [n_samples,n_steps,D_input],转换成[n_steps,n_samples,D_input
inputs_T = tf.transpose(re2, perm=[1, 0, 2])
# 实例rnn的正向cell，这里使用的是GRUcell
rnn_fcell = GRUcell(inputs_T, num_units, num_units, orthogonal_initializer)
# 实例rnn 的反向cell
rnn_bcell = GRUcell(inputs_T, num_units, num_units, orthogonal_initializer)
# 将两个cell 送个 scan 里计算，并使用sum的方式合并两个方向所计算的数据
rnn0 = RNN(rnn_fcell, rnn_bcell)
# 将输入数据从 [n_samples,n_steps,D_input],reshape成 [n_samples*n_steps,D_input]
# 用于feedforward layer 的使用
rnn1 = tf.reshape(rnn0, [-1, num_units])
# dropout
rnn2 = tf.nn.dropout(rnn1, drop_keep_rate)
# 第三层
W0 = weight_init([num_units, num_units])
b0 = bias_init([num_units])
rnn3 = tf.nn.relu(tf.matmul(rnn2, W0) + b0)
rnn4 = tf.nn.dropout(rnn3, drop_keep_rate)

# 第四层
W1 = weight_init([num_units, num_units])
b1 = bias_init([num_units])
rnn5 = tf.nn.relu(tf.matmul(rnn4, W1) + b1)
rnn6 = tf.nn.dropout(rnn5, drop_keep_rate)
# 输出层
W = weight_init([num_units, D_label])
b = bias_init([D_label])
output = tf.matmul(rnn6, W) + b

# loss
loss = tf.reduce_mean((output - labels) ** 2)
L2_total = tf.nn.l2_loss(Wf0) + tf.nn.l2_loss(Wf1) + tf.nn.l2_loss(W0) + tf.nn.l2_loss(W1) + tf.nn.l2_loss(
    W)  # + rnn_fcell.l2_loss+run_bcell.l2_loss
# 训练所需的
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss + L2_penalty * L2_total)

# 开始训练
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 训练并记录
def train_epoch(EPOCH):
    for k in range(EPOCH):
        train0 = shufflelists(train)
        for i in range(len(train)):
            rnn0_shape,re1_shape,inputs_shape,h2d_shape,re2_shape,inputs_T_shape,_ =sess.run([tf.shape(rnn0),tf.shape(re1),tf.shape(inputs),tf.shape(h2d),tf.shape(re2),tf.shape(inputs_T),train_step],feed_dict={inputs:train0[i][0],labels:train0[i][1],drop_keep_rate:0.7})
            print("re1.shape:", re1_shape)
            print("inputs.shape:", inputs_shape)
            print("h2d.shape:", h2d_shape)
            print("re2.shape:", re2_shape)
            print("inputs_T.shape:", inputs_T_shape)
            print("rnn0.shape:", rnn0_shape)

        tl = 0
        dl = 0
        for i in range(len(test)):
            dl += sess.run(loss,feed_dict={inputs:test[i][0],labels:test[i][1],drop_keep_rate:1.0})
        for i in range(len(train)):
            tl += sess.run(loss,feed_dict={inputs:train[i][0],labels:train[i][1],drop_keep_rate:1.0})
        print(k,"train:",round(tl/83,3),"test:",round(dl/20,3))

t0 = time.time()
train_epoch(100)
t1 = time.time()
print("%f seconds "%round(t1-t0,2))
