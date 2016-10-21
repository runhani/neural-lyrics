from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import pdb


import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

maxlen = 40

path = "data/korean_lyrics.txt"
text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


model = load_model('lyrics_model.h5')

input_string = raw_input("please put seed korean sentence: ")


prefix ="\n\n\n\n" + "01" + " "
surfix ="\n"

sentence = prefix + input_string

print sentence

padding = maxlen - len(sentence) - len(surfix)


for i in range(padding):
    sentence += str(' ')

sentence += surfix

print len(sentence)
# pdb.set_trace()

diversity = 0.4
print()
print('----- diversity:', diversity)

generated = ''

generated += sentence
print('----- Generating with seed: "' + sentence + '"')
sys.stdout.write(generated)

for i in range(400):
    x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()
print()