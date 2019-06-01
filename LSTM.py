import tensorflow as tf
import numpy as np

class LSTM(object):
    def __init__(self, num_hidden):
        self.cell_fw = tf.contrib.rnn.LSTMCell(num_hidden,
                  initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                  state_is_tuple=False)
        self.cell_bw = tf.contrib.rnn.LSTMCell(num_hidden,
                  initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
                  state_is_tuple=False)
        (emb_encoder_inputs, fw_state, _) = tf.contrib.rnn.static_bidirectional_rnn(
              cell_fw, cell_bw, emb_encoder_inputs, dtype=tf.float32,
              sequence_length=article_lens)
