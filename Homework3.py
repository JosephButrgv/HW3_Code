import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from keras.utils import to_categorical, pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer


# Load the data
df = pd.read_json(
    '/content/drive/MyDrive/VideoGames/Video_Games.json.gz', lines=True)

# Preprocess the data
df['reviewText'] = df['reviewText'].fillna(
    '')  # Replace NaN values with empty strings
tokenizer = Tokenizer(num_words=5000)  # Only keep the 5000 most common words
tokenizer.fit_on_texts(df['reviewText'])
sequences = tokenizer.texts_to_sequences(df['reviewText'])
# Pad sequences to a maximum length of 100
X = pad_sequences(sequences, maxlen=100)
y = np.array(df['overall']) - 1  # Convert the labels to integer format

# Split the data into training and testing sets
(X_train, X_test), (Y_train, Y_test) = train_test_split(
    X, y, test_size=0.2, random_state=42)

x_train = pad_sequences(X_train, maxlen=100)
x_test = pad_sequences(X_test, maxlen=100)
y_train = Y_train
y_test = Y_test

# check for out-of-range values in y_train
y_train_range = np.logical_or(y_train == 0, y_train == 1)
y_train[y_train_range == False] = 0

# check for out-of-range values in y_test
y_test_range = np.logical_or(y_test == 0, y_test == 1)
y_test[y_test_range == False] = 0

# convert labels to one-hot encoded vectors
y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)

model = Sequential()
model.add(Embedding(1000, 100))
model.add(LSTM(100, activation='tanh'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train_cat, batch_size=100,
                    epochs=40, validation_data=(x_test, y_test_cat))
print("\n test accuracy: %.4f" % (model.evaluate(x_test, y_test_cat)[1]))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test_cat)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
