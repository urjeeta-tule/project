from __future__ import print_function

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

from gensim.models import Word2Vec
import nltk
import numpy as np
from nltk.corpus import stopwords
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

nltk.download('stopwords')

start_time = time.time()

def pre_process_lines(text):
#	text = text.decode(encoding = 'UTF-8',errors = 'strict').split()
	text = text.split()
	punctuations=['@','#','$','%','^','&','*','(',')','-','_','+','=','{','}','[',']','|','\\\\','\"',"\'",';',':','<','>','/',',','.','?','!','\\n']
	
	for punctuation in punctuations:
		if punctuation in text:
			text.remove(punctuation)

	stopwords = nltk.corpus.stopwords.words('english')
	word_list = []   
	
	for w in text:
		if w not in stopwords:
			word_list.append(w)
	return word_list

def generate_word2vec_model(dataset):
	processed_sentences_list = []

	for data_point in dataset:
		processed_sentences_list.append(data_point['word_list'])

	processed_sentences_list = np.array(processed_sentences_list)
	print ("total examples %s" % len(processed_sentences_list))

	# model = Word2Vec(X, size=2, window=5, min_count=2, workers=1)
	model = Word2Vec(processed_sentences_list, size=100, window=5, min_count=2, workers=1)
	return model

def create_label(label_text):
	label_dict = {
		"ESFJ": [1,0,0,0,0], 
		"ESTJ": [0,1,0,0,0],
		"ISFJ": [0,0,1,0,0],
		"ISFP": [0,0,0,1,0],
		"ISTJ": [0,0,0,0,1]
	}
	value = np.array(label_dict[label_text])
	return value

def generate_dataset():

	train_dataset = []

	with open("mbtiv245.csv", "r", encoding="UTF-8") as infile:
		for line in infile:
			label_text, text = line.split("\t")
			label = create_label(label_text)
			word_list = pre_process_lines(text)
			data_point = {
				'word_list': word_list,
				'label': label 
			}
			
			train_dataset.append(data_point)
	
	return train_dataset

def train_neural_net(word2vec_model, dataset):
	sentence_vector_list = []
	sentence_labels = []

	for data_point in dataset:
		word_vector = np.array(
			np.mean(
				[word2vec_model[word] for word in data_point['word_list'] if word in word2vec_model]
				or
				[np.zeros(100)], axis=0
			)
		)
		
		sentence_vector_list.append(word_vector.reshape(len(word_vector)))
		sentence_labels.append(data_point['label'])

	return sentence_vector_list,sentence_labels
	
# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 800 # 1st layer number of features
n_hidden_2 = 400 # 2nd layer number of features
n_hidden_3 = 200
n_hidden_4 = 100
n_input = 100 # MNIST data input (img shape: 28*28)
n_classes = 5 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)
	layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
	layer_3 = tf.nn.relu(layer_3)
	layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
	layer_4 = tf.nn.relu(layer_4)
    # Output layer with linear activation
	out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
	return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
	'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
	'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	dataset = generate_dataset()
	word2vec_model = generate_word2vec_model(dataset)
	Xtr,ytr = train_neural_net(word2vec_model,dataset)
    # Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		#total_batch = 10000	#int(mnist.train.num_examples/batch_size)
		
		# Loop over all batches
        #for i in range(total_batch):
        #    batch_x, batch_y = mnist.train.next_batch(batch_size)
		#Run optimization op (backprop) and cost op (to get loss value)
		_, c = sess.run([optimizer, cost], feed_dict={x: Xtr,y: ytr})
            # Compute average loss
		#avg_cost += c / total_batch
        # Display logs per epoch step
		#if epoch % display_step == 0:
			#print("Epoch:", '%04d' % (epoch+1), "cost=", \"{:.9f}".format(avg_cost))
	print("Optimization Finished!")

    # Test model
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print("Accuracy:", accuracy.eval({x:Xtr, y:ytr}))