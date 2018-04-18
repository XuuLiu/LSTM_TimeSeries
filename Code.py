import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from scipy.stats import ks_2samp

filename='/Users/XuLiu/Desktop/Math/data/shenyangA.csv'
def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'r').read()
    data = f.split('\n')
    sum=0
    dataout=[]
    for a in data:
        sum=sum+int(a)
        dataout.append(int(a))
    average=sum/len(data)

    for i in range(12):
        data.append(str(average))
        i=i+1



    sequence_length = seq_len
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length+1])

    if normalise_window:
        result = normalise_windows(result)



    result = np.array(result)

    x = result[:, :-1]
    y = result[:, -1]

    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    dataout = np.array(dataout)

    return [dataout, x, y]


def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data


def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    #print("Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model,data):
    predicted=model.predict(data)
    #print('predicted shape:',np.array(predicted).shape)
    predicted=np.reshape(predicted,(predicted.size,))
    return predicted


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


epochs  = 10
seq_len = 24


dataorg, X, Y = load_data(filename, seq_len,True)


model = build_model([1, 12, 24, 1])

model.fit(
    X,
    Y,
    batch_size=24,
    epochs=epochs,
    validation_split=0.05)

point_by_point_predictions = predict_point_by_point(model, X)

plot_results(point_by_point_predictions,Y)
#print('KS:', ks_2samp(point_by_point_predictions, Y))

new_Y=Y[-12:]
learn_value=[]
pred_value=[]
for i in range(12):
    pred_value.append((new_Y[i]+1)*dataorg[24+i])
for i in range(12):
    learn_value.append((Y[i]+1)*dataorg[12+i])

