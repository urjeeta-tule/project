from gensim.models import Word2Vec
import nltk
import numpy as np
from nltk.corpus import stopwords
import time
from sklearn.neural_network import MLPClassifier

nltk.download('stopwords')

start_time = time.time()

def pre_process_lines(text):
	text = text.decode('utf-8').split()

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
		"ISFP": [1, 0, 0, 0, 0], 
		"ENTP": [0, 1, 0, 0, 0],
		"ESTP": [0, 0, 1, 0, 0],
		"INFP": [0, 0, 0, 1, 0],
		"ISFJ": [0, 0, 0, 0, 1]
	}
	value = np.array(label_dict[label_text])
	return value

def generate_dataset():

	train_dataset = []

	with open("mbti_2.csv", "r") as infile:
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

	clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(400, 400, 200, 100), random_state=1, max_iter=20000)
	clf.fit(sentence_vector_list, sentence_labels)
	return clf

dataset = generate_dataset()
word2vec_model = generate_word2vec_model(dataset)
trained_model = train_neural_net(word2vec_model, dataset)