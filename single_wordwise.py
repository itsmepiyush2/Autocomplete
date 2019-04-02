import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

text = """ Jack and Jill went up the hill\n
		To fetch a pail of water\n
		Jack fell down and broke his crown\n
		And Jill came tumbling after\n """

tokeniser = Tokenizer()
tokeniser.fit_on_texts([text])
encoded = tokeniser.texts_to_sequences([text])[0]

vocab_size = len(tokeniser.word_index) + 1

sequences = []
for i in range(1, len(encoded)):
    sequence = encoded[i-1 : i+1]
    sequences.append(sequence)

sequences = np.array(sequences)

x, y = sequences[:, 0], sequences[:, 1]

y = to_categorical(y, num_classes = vocab_size)
# in case of taking vocab_size = len(tokeniser.word_index) the following error appears
# "IndexError: index 21 is out of bounds for axis 1 with size 21"
# problem is solved on adding 1 to vocab_size

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length = 1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

model.fit(x, y, epochs = 350, verbose = 2)

def generate_seq(model, tokeniser, first_word, words):
	in_text, result = first_word, first_word
	for _ in range(words):
		encoded = tokeniser.texts_to_sequences([in_text])[0]
		encoded = np.array(encoded)
		y_pred = model.predict_classes(encoded, verbose=0)
		out_word = ''
		for word, index in tokeniser.word_index.items():
			if index == y_pred:
				out_word = word
				break
		in_text, result = out_word, result + ' ' + out_word
	return result

print(generate_seq(model, tokeniser, 'Jack', 13))