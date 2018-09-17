
# coding: utf-8

# In[1]:


import glob
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
from math import floor
import sys
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[8]:


def get_file_list_from_dir(parent_dir,sub_dirs, file_ext="*.wav"):
    all_files = []
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            all_files.append(fn)
    return random.sample(all_files,len(all_files))
            



def get_training_and_testing_sets(file_list):
    split = 0.7
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing



def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)
        
        
       # label = fn.split('/')[-1].split('.')[0].split('-')[1]
        
def extract_features(path_list,file_ext="*.wav",bands = 60, frames = 41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []

    for fn in path_list:
        sound_clip,s = librosa.load(fn)
        label = fn.split('/')[-1].split('.')[0].split('-')[1]
        for (start,end) in windows(sound_clip,window_size):
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                logspec = librosa.amplitude_to_db(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)
                labels.append(label)
            
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    
    return np.array(features), np.array(labels,dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(1.0, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,2,2,1], padding='SAME')

def apply_convolution(x,kernel_size,num_channels,depth):
    weights = weight_variable([kernel_size, kernel_size, num_channels, depth])
    biases = bias_variable([depth])
    return tf.nn.relu(tf.add(conv2d(x, weights),biases))

def apply_max_pool(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1], 
                          strides=[1, stride_size, stride_size, 1], padding='SAME')


# In[13]:


def ConeNect (parent_dir):
      
    sub_dirs = ['data1','data2','data3','data4','data5','data6','data7', 'data8', 'data9','data10']
    training, testing = get_training_and_testing_sets(get_file_list_from_dir(parent_dir,sub_dirs))
    print(len(training))
    print(len(testing))
	
    tr_features,tr_labels = extract_features(training)
    tr_labels = one_hot_encode(tr_labels)

    ts_features,ts_labels = extract_features(testing)
    ts_labels = one_hot_encode(ts_labels)

    

    frames = 41
    bands = 60

    feature_size = 2460 #60x41
    num_labels = 10
    num_channels = 2

    batch_size = 50
    kernel_size = 30
    depth = 20
    num_hidden = 200

    learning_rate = 0.01
    total_iterations = 2000

    X = tf.placeholder(tf.float32, shape=[None,bands,frames,num_channels])
    Y = tf.placeholder(tf.float32, shape=[None,num_labels])

    cov = apply_convolution(X,kernel_size,num_channels,depth)

    shape = cov.get_shape().as_list()
    cov_flat = tf.reshape(cov, [-1, shape[1] * shape[2] * shape[3]])

    f_weights = weight_variable([shape[1] * shape[2] * depth, num_hidden])
    f_biases = bias_variable([num_hidden])
    f = tf.nn.sigmoid(tf.add(tf.matmul(cov_flat, f_weights),f_biases))

    out_weights = weight_variable([num_hidden, num_labels])
    out_biases = bias_variable([num_labels])
    y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

    loss = -tf.reduce_sum(Y * tf.log(y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    cost_history = np.empty(shape=[1],dtype=float)
    with tf.Session() as session:
        tf.initialize_all_variables().run()

        for itr in range(total_iterations):    
            offset = (itr * batch_size) % (tr_labels.shape[0] - batch_size)
            batch_x = tr_features[offset:(offset + batch_size), :, :, :]
            batch_y = tr_labels[offset:(offset + batch_size), :]

            _, c = session.run([optimizer, loss],feed_dict={X: batch_x, Y : batch_y})
            cost_history = np.append(cost_history,c)
            
        print('Test accuracy: ',round(session.run(accuracy, feed_dict={X: ts_features, Y: ts_labels}) , 3))
        fig = plt.figure(figsize=(15,10))
        plt.plot(cost_history)
        plt.axis([0,total_iterations,0,np.max(cost_history)])
        plt.show()


# In[ ]:


if __name__ == '__main__':
    parent_dir = sys.argv[1]
    ConeNect(parent_dir)  

