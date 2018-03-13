import time
import numpy as np
import tensorflow as tf

# 加载数据
with open("anna.txt", 'r') as f:
    text = f.read()

# 构建字符集合
vocab = set(text)
# 字符-数字映射字典
vocab_to_int = {c: i for i, c in enumerate(vocab)}
# 数字-字符映射字典
int_to_vocab = dict(enumerate(vocab))
print("len of vocab:",len(vocab))

# 对文本进行转码
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)


def get_batches(arr, n_seqs, n_steps):
    '''
    对已有的数组进行mini-batch分割

    arr: 待分割的数组
    n_seqs：一个batch中的序列个数
    n_steps：单个序列的长度
    '''
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)

    # 这个地方我们保留完整的batch(full batch)，也就是说对于不能整除的部分进行舍弃
    arr = arr[:batch_size * n_batches]

    # 重塑
    arr = arr.reshape((n_seqs, -1))
    # print(arr)

    for n in range(0, arr.shape[1], n_steps):
        #         print("arr:",arr)
        x = arr[:, n: n + n_steps]
        y_tem = arr[:, n + 1:n + n_steps + 1]
        # 注意 targets相比于 x 会 向后错位一个字符
        y = np.zeros_like(x)
        y[:, :y_tem.shape[1]] = y_tem
        yield x, y


def build_inputs(num_seqs, num_steps):
    '''
    构建输入层

    num_seqs：每个batch中的序列个数
    num_steps：每个序列包含的字符数
    '''
    inputs = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='targets')

    # 加入 keep_prob
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return inputs, targets, keep_prob


def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    '''
    构建lstm层

    lstm_size: lstm cell 中隐藏节点的数目
    num_layers: lstm 层的数目
    batch_size： num_seqs * num_steps
    keep_prob：
    '''
    def get_drop(lstm_size, keep_prob):
        # 构建一个基本lstm单元
        lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

        # 添加 dropout
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop

    # 堆叠多个LSTM单元
    cell = tf.nn.rnn_cell.MultiRNNCell([get_drop(lstm_size,keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state


def build_output(lstm_output, in_size, out_size):
    '''
    构建输出层

    lstm_output:LSTM 层的输出结果（是一个三位数组）
    in_size: LSTM层重塑后的size
    out_size： softmax层的size
    '''
    #  将lstm的输出按照行 concat。例如：[[1,2,3],[7，8，9]]
    # tf.concat 的结果是 [1,2,3,7,8,9]
    seq_output = tf.concat(lstm_output, axis=1)
    # reshape
    x = tf.reshape(seq_output, [-1, in_size])

    # 连接LSTM输入到softmax layer
    with tf.variable_scope("softmax"):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))

    # 计算logits
    logits = tf.matmul(x, softmax_w) + softmax_b

    # softmax 层返回概率分布
    out = tf.nn.softmax(logits=logits, name="predictions")

    return out, logits


def build_loss(logits, targets, lstm_size, num_classes):
    '''
    根据logits和targets计算损失

    logits: 全连接层输出的结果（没有经过softmax）
    targets： 目标字符
    lstm_size： LSTM cell 隐藏层节点的数量
    num_classes: vocab_size
    '''
    # 对targets进行编码
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

    # softmax cross entropy between logits and labels
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)

    return loss


def build_optimizer(loss, learning_rate, grad_clip):
    '''
    构造Optimizer

    loss: 损失
    learing_rate： 学习率
    grad_clip: 修剪的阈值
    '''
    # 使用clipping gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return optimizer


class CharRNN(object):
    def __init__(self, num_classes, batch_size=64, num_steps=50, lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False):
        # 如果 sampling是 True，则采用 SGC
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()

        # 输入层
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

        # LSTM 层
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        # 对输入进行 one-hot 编码
        self.x_one_hot = tf.one_hot(self.inputs, num_classes)
        print('num_classes:',num_classes)

        # 运行 RNN
        self.outputs, self.state = tf.nn.dynamic_rnn(cell=cell, inputs=self.x_one_hot, initial_state=self.initial_state)
        self.final_state = self.state

        # 预测结果
        self.prediction, self.logits = build_output(self.outputs, lstm_size, num_classes)

        # loss 和 optimizer(with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)


# 开始训练
batch_size = 100
num_steps = 100
lstm_size = 512
num_layers = 2
learning_rate = 0.001
keep_prob = 0.5

epochs = 2
sava_every_n = 200  # 每n轮进行一次变量保存

model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps, lstm_size=lstm_size, num_layers=num_layers,
                learning_rate=learning_rate)

saver = tf.train.Saver(max_to_keep=100)

sess = tf.InteractiveSession()
# with tf.Session() as sess:
sess.run(tf.global_variables_initializer())

counter = 0
for e in range(epochs):
    # Train network
    new_state = sess.run(model.initial_state)
    loss = 0
    for x, y in get_batches(encoded, batch_size, num_steps):
        counter += 1
        start = time.time()
        feed = {
            model.inputs: x,
            model.targets: y,
            model.keep_prob: keep_prob,
            model.initial_state: new_state
        }
        batch_loss, new_state, _ = sess.run([
            model.loss, model.final_state, model.optimizer
        ], feed_dict=feed)
        end = time.time()
        # control the print lines
        if counter % 100 == 0:
            print("轮数：{}/{}...".format(e + 1, epochs),
                  "训练步数：{}...".format(counter),
                  '训练误差：{:.4f}...'.format(batch_loss),
                  '{:.4f} sec/batch'.format((end - start))
                  )

        # if (counter % sava_every_n == 0):
        #     saver.save(sess, "checkpoints/i{}_1{}.ckpt".format(counter, lstm_size))
