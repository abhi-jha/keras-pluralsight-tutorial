from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt
import os

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def plot_data(p1, X, y):
    p1.plot(X[y == 0, 0], X[y == 0, 1], 'ob', alpha=0.5,  color='red')

    p1.plot(X[y == 1, 0], X[y == 1, 1], 'xr', alpha=0.5,  color='green')

    p1.legend(['0', '1'])

    return p1

def plot_decison_boundary(model, X, y):

    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)

    aa, bb = np.meshgrid(hticks, vticks)

    ab = np.c_[aa.ravel(), bb.ravel()]

    c = model.predict(ab)
    Z= c.reshape(aa.shape)

    plt.figure(figsize=(12,8))

    plt.contour(aa, bb, Z, cmap = 'bwr', alpha = 0.2)

    plot_data(plt, X, y)

    return plt


X, y = make_circles(n_samples=1000, factor=.6 , noise=0.1, random_state=42)

p1 = plot_data(plt,X, y)

p1.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train , y_test = train_test_split(X, y , test_size=0.3, random_state=42)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model = Sequential()

model.add(Dense(4, input_shape=(2,), activation='tanh', name='Hidden-1'))

model.add(Dense(4, activation='tanh', name='Hidden-2'))

model.add(Dense(1, activation='sigmoid', name='Output_layer'))

model.summary()

model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])

from keras.utils import plot_model

plot_model(model, to_file='model.png',show_shapes=True, show_layer_names=True, rankdir='LR')

model.fit(X_train, y_train, epochs=1000, verbose=1)

eval_result = model.evaluate(X_test, y_test)

print("\n\nTest loss : " ,eval_result[0], "  Test accuracy  : ", eval_result[1])

plot_decison_boundary(model, X, y).show()

