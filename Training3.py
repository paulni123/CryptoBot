from CoinbaseClient import CoinbaseClient
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import calendar
import csv

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


def retrieveData():

    testClient = CoinbaseClient()
    #testClient.getCoinHistoricalData('BTC-USD',1625616000,1625702399, 300)

    #start_time = datetime.fromtimestamp(datetime(2021, 7, 1).timestamp())

    # with open('testData3/' + start_time.strftime('%m-%d-%y') +'.json', 'w') as f:
    #     f.write(json.dumps(testClient.getCoinHistoricalData('BTC-USD',start_time,end_time, 300)))

    start_time = datetime(2020,12,1)
    start_time_timestamp = start_time.timestamp()
    end_time = datetime(2020, 12, 2)
    end_time_timestamp = end_time.timestamp()

    numOfDays = calendar.monthrange(start_time.year, start_time.month)[1]

    tempList = []

    with open('testData3/json/2020/' + start_time.strftime('%m-%Y') +'.json', 'w') as f:
        for _ in range(numOfDays):
            tempList += testClient.getCoinHistoricalData('BTC-USD',start_time,end_time, 300)[::-1]

            start_time_timestamp += 86400
            start_time = datetime.fromtimestamp(start_time_timestamp)
            end_time_timestamp += 86400
            end_time = datetime.fromtimestamp(end_time_timestamp)
        
        f.write(json.dumps(tempList))


    #Mon Jan 19 14:10:20 1970


def temp():
    with open('testData3/json/2020/12-2020.json') as json_file:
        jsonData = json.load(json_file)

    trainingCSV = open('testData3/csv/2020/12-2020.csv', 'w', newline='')
    csv_writer = csv.writer(trainingCSV)

    for data in jsonData:
        csv_writer.writerow(data)
    trainingCSV.close()


# retrieveData()
# temp()



NUMERIC_COLUMNS = ["Time", "Low", "High", "Open", "Close", "Volume"]

trainingData = pd.read_csv('testData3/csv/2021/01-2021.csv', names=NUMERIC_COLUMNS)
print(trainingData)

trainingData = trainingData.iloc[:, 4].values  #Close values
print(type(trainingData))

print(trainingData)


#Normalization Function [x_new = (x_i - min(x))/(max(x) - min(x))]

scaler = MinMaxScaler()

trainingData = scaler.fit_transform(trainingData.reshape(-1,1))

x_trainingdata = []
y_trainingdata = []

for i in range(40, len(trainingData)):
    x_trainingdata.append(trainingData[i-40:i,0])
    y_trainingdata.append(trainingData[i,0])



x_trainingdata = np.array(x_trainingdata)
y_trainingdata = np.array(y_trainingdata)

print(x_trainingdata.shape)

print(y_trainingdata.shape)


x_trainingdata = np.reshape(x_trainingdata, (x_trainingdata.shape[0], x_trainingdata.shape[1], 1))



rnn = Sequential()

rnn.add(LSTM(units=45,return_sequences=True,input_shape = (x_trainingdata.shape[1], 1)))
rnn.add(Dropout(0.2))

for i in [True,True,False]:
    rnn.add(LSTM(units=45,return_sequences=i))
    rnn.add(Dropout(0.2))


rnn.add(Dense(units=1))

rnn.compile(optimizer='adam',loss='mean_squared_error')

rnn.fit(x_trainingdata, y_trainingdata, epochs = 10, batch_size = 32)


testdata = pd.read_csv('testData3/csv/2021/02-2021.csv', names=NUMERIC_COLUMNS)
testdata = testdata.iloc[:,4].values

print(testdata.shape)

plt.plot(testdata)
plt.show()

unscaled_trainingdata = pd.read_csv('testData3/csv/2021/01-2021.csv', names=NUMERIC_COLUMNS)
unscaled_testdata = pd.read_csv('testData3/csv/2021/02-2021.csv', names=NUMERIC_COLUMNS)

combinedData = pd.concat((unscaled_trainingdata['Open'], unscaled_testdata['Open']), axis=0)

print(combinedData.shape)

x_testdata = combinedData[len(combinedData) - len(testdata) - 40:].values

x_testdata = np.reshape(x_testdata, (-1,1))

x_testdata = scaler.transform(x_testdata)

final_x_testdata = []

for i in range(40, len(x_testdata)):
    final_x_testdata.append(x_testdata[i-40:i,0])

final_x_testdata = np.array(final_x_testdata)

final_x_testdata = np.reshape(final_x_testdata, (final_x_testdata.shape[0], 

                                               final_x_testdata.shape[1], 

                                               1))


predictions = rnn.predict(final_x_testdata)

unscaled_predictions = scaler.inverse_transform(predictions)

plt.clf()
plt.plot(unscaled_predictions, color = '#135485', label = "Predictions")
plt.plot(testdata, color = 'black', label = "Real Data")
plt.legend(loc = 'best')
plt.show()