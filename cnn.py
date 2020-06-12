# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
tf.__version__

# Importing the dataset
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

# Splitting the dataset to get the categorical variable
X_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1:].values
y_test = test.iloc[:, 0].values

# Reshaping the independent values into 28x28 pixel groups
X_train = np.array([np.reshape(i, (28,28)) for i in X_train])
X_test = np.array([np.reshape(i, (28,28)) for i in X_test])

# Reshaping the dependant variable
y_train = np.array(y_train).reshape(-1)
y_test = np.array(y_test).reshape(-1)

# Encoding categorical data
y_train = tf.keras.utils.to_categorical(y_train, 26)
y_test = tf.keras.utils.to_categorical(y_test, 26)

X_train = X_train.reshape((27455, 28, 28,1))
X_test = X_test.reshape((7172, 28, 28,1))

# Building the CNN

# Initialising the CNN
classifier = tf.keras.models.Sequential()

# Step 1 - Convolution
classifier.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[28, 28, 1]))

# Step 2 - Pooling
classifier.add(tf.keras.layers.MaxPool2D(pool_size=2))

# Adding a second convolutional layer
classifier.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
classifier.add(tf.keras.layers.MaxPool2D(pool_size=2))

# Adding a third convolutional layer
classifier.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
classifier.add(tf.keras.layers.MaxPool2D(pool_size=2))

# Adding a forth convolutional layer
classifier.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
classifier.add(tf.keras.layers.MaxPool2D(pool_size=2))

# Adding Dropout
classifier.add(tf.keras.layers.Dropout(0.2))

# Step 3 - Flattening
classifier.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
classifier.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
classifier.add(tf.keras.layers.Dense(units=26, activation='softmax'))

# Training the CNN

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set
history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size = 64, epochs = 50, verbose=0)

# Evaluating the accuracy
accuracy = classifier.evaluate(X_test, y_test)
print("Accuracy: ", accuracy[1])

# Saving the CNN Model
classifier.save('CNN_Model.h5')

# Predicting Test set results
y_pred = classifier.predict(X_test)

classifier.summary()



plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
