from __future__ import absolute_import,division,print_function,unicode_literals

import tensorflow as tf

mnist = tf.keras.datasets.mnist
#Tuple , first tuple contains training data and label and second tuple 
# contains test data 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#normalization for faster conversion 
x_train, x_test = x_train / 255.0, x_test / 255.0

#Sequential model 
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#train the model
model.fit(x_train, y_train, epochs=5)
#evaluate the accuracy
model.evaluate(x_test, y_test)
