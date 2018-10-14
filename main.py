import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# BUILD AND TRAIN MODEL
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

eval_stat = model.evaluate(x_test, y_test)
print('model test results:', eval_stat)

predict_dist = model.predict([x_test])
predicts = []
for e in predict_dist:
    predicts.append(np.argmax(e))

diff = list(np.array(predicts)-np.array(y_test))
i_of_mistakes = [i for i, e in enumerate(diff) if e != 0]

# MODEL USE
INDEX_OF_MISTAKE = 0

i = i_of_mistakes[INDEX_OF_MISTAKE]
print('prediction =', predicts[i], 'real =', y_test[i])
plt.imshow(x_test[i], cmap='binary')
plt.show()

img = Image.open('img/img11.png').convert('L')
arr = np.array(img) / 255
plt.imshow(arr, cmap='binary')
plt.show()
arr = np.array([arr])
p = model.predict(arr)
print('prediction:', np.argmax(p[0]))
print('distribution:', p[0])
