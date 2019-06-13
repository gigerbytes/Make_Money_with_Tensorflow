from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import numpy as np
import lstm, time #helper libraries
# Using TensorFlow backend.

#Step 1 Load Data
X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', 50, True)

#Step 2 Build Model
model = Sequential()

model.add(LSTM(
    input_dim=1,
    output_dim=50,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
print('compilation time : ', time.time() - start)

#Step 3 Train the model
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    batch_size=512,
    nb_epoch=2)
print(model.summary())


model.save("lstm.hdf5")

print(X_test)
print(y_test)#Step 4 - Plot the predictions!
predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)
print(predictions)
# lstm.plot_results_multiple(predictions, y_test, 50)
