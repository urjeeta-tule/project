from gensim.models import Word2Vec
import nltk
import numpy as np
from nltk.corpus import stopwords
import time
import random
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot
from sklearn.decomposition import PCA
from nltk.stem.snowball import SnowballStemmer


stemmer = SnowballStemmer("english")

start = time.time()

def pre_process_lines(text):
	text = text.decode('utf-8').lower().split()

	punctuations=['@','#','$','%','^','&','*','(',')','-','_','+','=','{','}','[',']','|','\\\\','\"',"\'",';',':','<','>','/',',','.','?','!','\\n']
	
	for punctuation in punctuations:
		if punctuation in text:
			text.remove(punctuation)

	stopwords = nltk.corpus.stopwords.words('english')
	word_list = []   
	
	for w in text:
		w = stemmer.stem(w)
		if w not in stopwords:
			word_list.append(w)
	return word_list


def generate_word2vec_model(dataset):
	processed_sentences_list = []

	for data_point in dataset:
		processed_sentences_list.append(data_point['word_list'])

	processed_sentences_list = np.array(processed_sentences_list)
	print ("Total number of datapoints : %s" % len(processed_sentences_list))

	# model = Word2Vec(X, size=2, window=5, min_count=2, workers=1)
	model = Word2Vec(processed_sentences_list, size=100, window=5, min_count=2, alpha=0.04)
	
	v = model.wv.vocab
	W = model[v]

	xount = 0
	s = ""
	for i in W:
		s = s + str(i)+"\n"
		xount += 1

	with open('vectors.txt', 'w') as the_file:
    		the_file.write(s)
    		
	print ("Total number of words : %s" % xount)

	return model

def plot(model):

	v = model.wv.vocab
	W = model[v]

	s=""
	for i in W[:]:
		s = s + str(i[0])+", "+str(i[1])+"\n"

	with open('vectors.txt', 'w') as the_file:
    		the_file.write(s)

	pca = PCA(n_components=2)
	result = pca.fit_transform(W)

	pyplot.scatter(result[:, 0], result[:, 1])

	s=""
	for i in result[:]:
		s = s + str(i[0])+", "+str(i[1])+"\n"

	with open('sample.txt', 'w') as the_file:
    		the_file.write(s)

	words = list(v)

	for i, word in enumerate(words):
		pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
	pyplot.show()


def create_label(label_text):
	label_dict = {
		"ISFP": [1, 0, 0, 0, 0], 
		"ESFJ": [0, 1, 0, 0, 0],
		"ESTJ": [0, 0, 1, 0, 0],
		"ISTJ": [0, 0, 0, 1, 0],
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


	return sentence_vector_list,sentence_labels

def predict_label(dataset,trained_model,word2vec_model):
	count = 0
	datapoint = dataset[44]
	actual_label = datapoint['label']
	word_vector = np.array(
	np.mean(
		[word2vec_model[word] for word in datapoint['word_list'] if word in word2vec_model]
		or 
		[np.zeros(100)], axis=0
		)
	)
	predicted_label = trained_model.predict(word_vector.reshape(1,-1))
	if np.array_equal(actual_label,predicted_label[0]):
		count += 1
	if count==1:
		print "Success!"
		print (" Actual label = %s " % actual_label)
		print ("Predicted label = %s " % predicted_label)
	else:
		print "Failure."
		print (" Actual label = %s " % actual_label)
		print ("Predicted label = %s " % predicted_label)
	
def cross_eval(word2vec_model,dataset):

	X,y = train_neural_net(word2vec_model,dataset)
	clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(800, 400, 200, 100), random_state=1, max_iter=5000)
	
	split = 0.6
	index = int(split*len(dataset))

	l = []

	for i in range(index):
 		n = random.randint(0,len(dataset))
 		while n in l:
 			n = random.randint(0,len(dataset))
 		l.append(n) 

 	Xtrain = []
	ytrain = []

	Xtest = []
	ytest = []


 	for i in range(len(dataset)):
 		if i in l:	
 			Xtrain.append(X[i])
 			ytrain.append(y[i])
 		else:
 			Xtest.append(X[i])
 			ytest.append(y[i])
	

	clf.fit(Xtrain,ytrain)

	tot_count = 0
	count = 0
	for i in range(len(Xtest)):
		actual_label = ytest[i]
		predicted_label = clf.predict(Xtest[i].reshape(1,-1))
		if np.array_equal(actual_label,predicted_label[0]):
			count += 1
 		tot_count += 1

 	print ("Out of %s samples " % tot_count)
 	print "Samples predicted correctly : "+str(count)
 	print "Success rate = "+str(float((float(count)/float(tot_count)))*100.0)


def testit(word2vec_model,trained_model):

	with open("test.txt","r") as afile:
		print "Sample text is :"
		print afile.read()
		word_list = pre_process_lines(afile.read())
		datapoint = {
			'word_list': word_list,
			'label': [0, 0, 0, 0, 0]
		}

	word_vector = np.array(
	np.mean(
		[word2vec_model[word] for word in datapoint['word_list'] if word in word2vec_model]
		or 
		[np.zeros(100)], axis=0
		)
	)

	predictd = trained_model.predict(word_vector.reshape(1,-1))

	return predictd[0]	

dataset = generate_dataset()
word2vec_model = generate_word2vec_model(dataset)
#plot(word2vec_model)
Xtr, Ytr = train_neural_net(word2vec_model, dataset)

trained_model = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(800, 400, 200, 100), random_state=1, max_iter=20000)
trained_model.fit(Xtr, Ytr)

cross_eval(word2vec_model,dataset)		
predict_label(dataset,trained_model,word2vec_model)

print testit(word2vec_model,trained_model)," predicted label"
finish = time.time()
print "Time taken in seconds : "+str(finish-start)

