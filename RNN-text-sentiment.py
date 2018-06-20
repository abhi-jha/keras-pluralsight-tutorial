#LSTM

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping
from keras.datasets import imdb

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

NUM_WORDS = 6000
SKIP_TOP = 0
MAX_REVIEW_LEN = 400

(x_train, y_train) , (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS, skip_top=SKIP_TOP)

#print("Encoded word sequence : ", x_train[3])

x_train = sequence.pad_sequences(x_train, maxlen=MAX_REVIEW_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_REVIEW_LEN)

print('x_train.shape : ', x_train.shape, ' x_test.shape : ', x_test.shape)

model = Sequential()
model.add(Embedding(NUM_WORDS, 64))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

BATCH_SIZE = 24
EPOCHS = 5

cbk_early_stopping = EarlyStopping(monitor='val_acc', mode='max')

model.fit(x_train, y_train, BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=[cbk_early_stopping])