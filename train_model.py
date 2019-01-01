import numpy as np
import plistlib
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# Global file names
in_filename = 'sequences/guns.txt'
model_name = 'models/model-guns.h5'
tokenizer_name = 'tokenizers/tokenizer-guns'

# load
doc = load_doc(in_filename)
lines = doc.splitlines()
lines_array = [s.split() for s in lines]

inputlen = len(lines_array[0]) - 1
print("inputlen: " + str(inputlen))

# integer encode sequences of words ONLY WORKS WITH KERAS 2.2 (coreml works with 2.1.6)
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

#words = sorted(list(set(doc.split())))
#tokenizer = dict((c, i) for i, c in enumerate(words))
#sequences = [[tokenizer[word] for word in line] for line in lines_array]

# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print("vocab size: " + str(vocab_size))

for l in lines:
    if not len(l.split()) == inputlen + 1:
        print("WRONG LENGTH: " + l + " " + str(len(l.split())))

# separate into input and output
sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

print("seq_length: " + str(seq_length))

# save the tokenizer
dump(tokenizer, open(tokenizer_name + '.pkl', 'wb'))
plistlib.dump(tokenizer.word_index, open(tokenizer_name + '.plist', 'wb'))

# define model
model = Sequential()
model.add(Embedding(vocab_size, inputlen, input_length=seq_length))
model.add(LSTM(200, return_sequences=True))
model.add(LSTM(200))
model.add(Dense(200, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=128, epochs=100)

# save the model to file
model.save(model_name)
