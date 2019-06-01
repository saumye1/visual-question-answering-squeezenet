import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
from clsidx_to_labels import labelForClassId
from squeezenet import SqueezeNet
from extract_data import load_data, load_images
from words_to_idx import wordsToIdx
from idx_to_words import idxToWords

SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PATH_TO_SAVED_NET = '/home/saumye/Documents/CS231N-stanford/assignment3/cs231n/datasets/squeezenet.ckpt'
hps = {
    'num_hidden' : 1024,
    'emb_dim' : 300,
    'batch_size' : 50,
    'vsize' : 11429, #number of distinct words in questions in all
    'classes' : 14008, #total number of distinct answers
    'enc_timesteps' : 23, #maximum words in a sentence
    'feature_layer' : 8
}

def question_to_vector(question, words_to_idx, timesteps):
    words_seq = question.split(" ")
    return [words_to_idx[word] for word in words_seq] + [0 for i in range(timesteps - words_seq.__len__())]

def preprocess_image(img):
    """Preprocess an image for squeezenet.

    Subtracts the pixel mean and divides by the standard deviation.
    """
    img = imresize(arr=img, size=(266, 400))
    return (img.astype(np.float32)/255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD

def LSTM(questions, question_lens, hps):
    encoder_inputs = tf.unstack(tf.transpose(questions))

    embedding = tf.get_variable('embedding', [hps['vsize'], hps['emb_dim']], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=1e-4))
    
    emb_encoder_inputs = [tf.nn.embedding_lookup(embedding, x) for x in encoder_inputs]

    cell_fw = tf.contrib.rnn.LSTMCell(hps['num_hidden'],
            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
            state_is_tuple=True)

    (emb_encoder_outputs, fw_state) = tf.contrib.rnn.static_rnn(cell_fw, emb_encoder_inputs,
                                    dtype=tf.float32, sequence_length=question_lens)
    return (emb_encoder_outputs, fw_state)

#load CNN
sess = tf.InteractiveSession()
CNN = SqueezeNet(save_path=PATH_TO_SAVED_NET, sess=sess)



#load question answer data
val_data = load_data()

print("=========== Data loaded ===========")

#process loaded data
# words = [set(sample['question'].split(" ")) for sample in val_data]
# set_words = set()
# for s in words:
#     set_words = set_words.union(s)

# hps['vsize'] = set_words.__len__()
words_to_idx = wordsToIdx()
idx_to_words = idxToWords()

hps['vsize'] = words_to_idx.__len__()
print("=========== Data Processed ============")

#give number to each word
# words_to_idx = { set_words.pop() : i for i in range(hps['vsize']) }
# idx_to_words = { words_to_idx[k] : k for k in words_to_idx.keys() }


answer_space = set(sample['answer'] for sample in val_data)
answer_to_idx = dict()
i = 0
while answer_space:
    answer_to_idx[answer_space.pop()] = i
    i = i + 1

idx_to_answer = { answer_to_idx[key] : key for key in answer_to_idx.keys()}

# variables to be defined: batch_size, enc_timesteps, vsize(vocabulary Size)
questions = tf.placeholder(tf.int32, [hps['batch_size'], hps['enc_timesteps']], name='questions')
question_lens = tf.placeholder(tf.int32, [hps['batch_size']], name='question_lens')

epochs = val_data.__len__() / hps['batch_size']
# epochs = 1



for i in range(epochs):
    print("========== Running Epoch {} / {} =========".format(i, epochs))
    test_data = val_data[i * hps['batch_size']:(i + 1) * hps['batch_size']]
    image_ids = [ sample['image_id'] for sample in test_data ]
    images_data = load_images(image_ids)

    print("\t {} Images loaded".format(images_data.__len__()))

    images_modified = [preprocess_image(im) for im in images_data]

    features = [CNN.extract_features(input=tf.reshape(im, [1, 266, 400, 3]))[hps['feature_layer']] for im in images_modified]
    features = [tf.nn.max_pool(feat, [1, 4, 4, 1], [1, 4, 4, 1], 'SAME') for feat in features]
    features = tf.stack(features, axis = 0)
    print("\t cnn features shape :::{}".format(features.shape))
    features = tf.reshape(features, (features.shape[0], -1))

    questions_vec = [ np.array( question_to_vector(test_data[j]['question'], words_to_idx, hps['enc_timesteps']) ) for j in range(hps['batch_size']) ]
    questions_vec = np.stack(questions_vec, axis = 0)
    questions_vec_lens = [sample['question'].split(" ").__len__() for sample in test_data]

    ground_truth = [ answer_to_idx[sample['answer']] for sample in test_data]
    one_hot_ground_truth = np.zeros([hps['batch_size'], hps['classes']])
    one_hot_ground_truth[np.arange(hps['batch_size']), ground_truth] = 1
    
    if i == 0:
        encoder_inputs = tf.unstack(tf.transpose(questions))

        embedding = tf.Variable(tf.random_uniform([hps['vsize'], hps['emb_dim']], -1.0, 1.0), name='embedding')
        # embedding = tf.get_variable('embedding', [hps['vsize'], hps['emb_dim']], dtype=tf.float32,
                    # initializer=tf.truncated_normal_initializer(stddev=1e-4))
        
        emb_encoder_inputs = [tf.nn.embedding_lookup(embedding, x) for x in encoder_inputs]

        cell_fw = tf.contrib.rnn.LSTMCell(hps['num_hidden'],
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                state_is_tuple=True)

        (emb_encoder_outputs, lstm_state) = tf.contrib.rnn.static_rnn(cell_fw, emb_encoder_inputs,
                                        dtype=tf.float32, sequence_length=question_lens)
        # (emb_encoder_outputs, lstm_state) = LSTM(questions, question_lens, hps)

        concatenated_features = tf.concat([features, lstm_state[1]], axis = 1)
        print("\t concatenated feature vector shape = {}".format(concatenated_features.shape))

        fc = tf.contrib.layers.fully_connected(concatenated_features, hps['classes'])
        print("\t fully connected layer shape = {}".format(fc.shape))
        # scores = tf.nn.softmax(fc)
        # predictions = tf.argmax(scores)

        losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_ground_truth, logits=fc)
        loss = tf.reduce_mean(losses)
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
        
        tf.initialize_all_variables().run()
            
    _, loss_value = sess.run([train_op, loss], feed_dict={questions: questions_vec, question_lens: questions_vec_lens})
    print "\t:::::: loss = {} train_op = {}".format(loss_value, _)




# run an image through CNN then contenate its one layer with the output of LSTM, i.e fw_state
# make a neural network, run through this and classify
# sess.run(tf.global_variables_initializer())
# _, lstm_state = sess.run([emb_encoder_outputs, lstm_state], feed_dict={questions : questions_vec, question_lens: questions_vec_lens})

print("lstm state variable = ", lstm_state, " shape is ", lstm_state[1].shape)

testCNN = False

if testCNN == True:
    test_image = imread('/home/saumye/Documents/CS231N-stanford/assignment3/kitten.jpg')
    plt.imshow(test_image)
    test_image = preprocess_image(test_image)
    avgPoolLayer, class_lb = CNN.sample(test_image, sess)
    print "avg pool layer shape = ", avgPoolLayer.shape
    print "The given image contains : ", labelForClassId(np.argmax(avgPoolLayer))
    plt.show()
