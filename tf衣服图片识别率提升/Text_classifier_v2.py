import tensorflow as tf
from tensorflow import keras


def decode_review(texts):
    return ' '.join([reverse_word_index.get(i, '?') for i in texts])

imdb = keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
print(x_train[0])

word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
print(decode_review(x_train[0]))

train_data = keras.preprocessing.sequence.pad_sequences(x_train,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(x_test,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# norm化
# train_data = tf.nn.l2_normalize(train_data.astype('float'))
# test_data = tf.nn.l2_normalize(test_data.astype('float'))

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim=10000, output_dim=16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

loss_and_metrics = model.evaluate(test_data, y_test)
print(loss_and_metrics)
