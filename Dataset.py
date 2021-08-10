from datetime import datetime
from keras.utils.generic_utils import populate_dict_with_module_objects
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.shape_base import block
from numpy.lib.financial import nper
from pandas import read_csv, DataFrame
import math
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from Constants import *


class Dataset:

    NUMERIC_COLUMNS = ["Time", "Low", "High", "Open", "Close", "Volume"]

    def __init__(self):
        self.featuresNames = []

    # def create_dataset(dataset, look_back=1):
    #     dataX, dataY = [], []
    #     for i in range(len(dataset) - look_back -1):
    #         temp = dataset[i:(i + look_back), 0]
    #         dataX.append(temp)
    #         dataY.append(dataset[i + look_back, 0])
    #     return np.array(dataX), np.array(dataY)

    # def train(self):
    #     dataframe = read_csv('testData3/csv/trainingData.csv', usecols=[3], engine='python')
    #     # dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
    #     dataset = dataframe.values
    #     dataset = dataset.astype('float32')
    #     # normalize the dataset
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     dataset = scaler.fit_transform(dataset)
    #     # split into train and test sets
    #     train_size = int(len(dataset) * 0.67)
    #     test_size = len(dataset) - train_size
    #     train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    #     # reshape into X=t and Y=t+1
    #     look_back = 3
    #     trainX, trainY = Dataset.create_dataset(train, look_back)
    #     testX, testY = Dataset.create_dataset(test, look_back)
    #     # reshape input to be [samples, time steps, features]
    #     trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    #     testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    #     # create and fit the LSTM network
    #     model = Sequential()
    #     model.add(LSTM(4, input_shape=(1, look_back)))
    #     model.add(Dense(1))
    #     model.compile(loss='mean_squared_error', optimizer='adam')
    #     model.fit(trainX, trainY, epochs=100, batch_size=3000)
    #     # make predictions
    #     trainPredict = model.predict(trainX)
    #     testPredict = model.predict(testX)
    #     # invert predictions
    #     trainPredict = scaler.inverse_transform(trainPredict)
    #     trainY = scaler.inverse_transform([trainY])
    #     testPredict = scaler.inverse_transform(testPredict)
    #     testY = scaler.inverse_transform([testY])
    #     # calculate root mean squared error
    #     trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    #     print('Train Score: %.2f RMSE' % (trainScore))
    #     testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    #     print('Test Score: %.2f RMSE' % (testScore))
    #     # shift train predictions for plotting
    #     trainPredictPlot = np.empty_like(dataset)
    #     trainPredictPlot[:, :] = np.nan
    #     trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    #     # shift test predictions for plotting
    #     testPredictPlot = np.empty_like(dataset)
    #     testPredictPlot[:, :] = np.nan
    #     testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    #     # plot baseline and predictions
    #     plt.plot(scaler.inverse_transform(dataset))
    #     plt.plot(trainPredictPlot)
    #     plt.plot(testPredictPlot)
    #     plt.show()


    def loadCryptoData(self, months):
        cryptoData = []

        for month in months:
            year = datetime.strptime(month, '%m-%Y').strftime('%Y')
            with open(DatasetPath + year + '/'+ month + '.json') as jsonFile:
                jsonData = json.load(jsonFile)

                for data in jsonData:
                    if len(data) != 0:
                        cryptoData.append(data)

        return cryptoData
            
    
    # NUMERIC_COLUMNS = ["Time", "Low", "High", "Open", "Close", "Volume"]


    # def createDataset(self, cryptoData, isTraining, training_window = 180, labeling_window = 60, feature_window = 30):

    #     x = []
    #     y = []

    #     prices = []

    #     if isTraining:
    #         # btcData = read_csv('testData3/csv/trainingData.csv', names= Dataset.NUMERIC_COLUMNS)
    #         btcData = read_csv('testData3/csv/2021/01-2021.csv', names= Dataset.NUMERIC_COLUMNS)
    #     else:
    #         btcData = read_csv('testData3/csv/2021/02-2021.csv', names = Dataset.NUMERIC_COLUMNS)

        

    #     features = []
    #     self.featuresNames = []

    #     buying = 0
    #     selling = 0


    #     #Signals and moving averages

    #     shortWindow = 40
    #     longWindow = 100

    #     signals = DataFrame(index=btcData.index)
    #     signals['Signal'] = 0.0


    #     signals['short_mavg'] = btcData['Close'].rolling(window=shortWindow, min_periods=1, center=False).mean()

    #     signals['long_mavg'] = btcData['Close'].rolling(window=longWindow, min_periods=1, center=False).mean()

    #     signals['Position'] = 0.0

    #     #Features and Feature Names

    #     prevSignal = 0.0
    #     position = 0.0



    #     startIndex = 0
    #     endIndex = training_window -1
    #     block = 60

    #     for i in range(startIndex, endIndex):

    #         #if i % 50000 == 0:
    #         if i % 1000 == 0:
    #             print("Checkpoint at: " + str(i))

    #         cur_low = cryptoData[i][1]
    #         cur_high = cryptoData[i][2]
    #         cur_open = cryptoData[i][3]
    #         cur_close = cryptoData[i][4]
    #         cur_volume = cryptoData[i][5]



    #         if i >= shortWindow:
                
    #             #Creating signal when short moving average crosses long moving average
    #             if signals['short_mavg'][i] > signals['long_mavg'][i]:
    #                 signals['Signal'][i] = 1.0
    #                 position = 1.0 - prevSignal
    #                 signals['Position'][i] = position
    #                 prevSignal = 1.0
    #             else:
    #                 signals['Signal'][i] = 0.0
    #                 position = 0.0 - prevSignal
    #                 signals['Position'][i] = position
    #                 prevSignal = 0.0
        

    #         self.featuresNames.append("Low_Price")
    #         features.append(cur_low)
    #         self.featuresNames.append("High_Price")
    #         features.append(cur_high)
    #         self.featuresNames.append("Open_Price")
    #         features.append(cur_open)
    #         self.featuresNames.append("Volume")
    #         features.append(cur_volume)

    #         self.featuresNames.append("Close_Price")
    #         features.append(cur_close)
    #         prices.append(cur_close)

    #         self.featuresNames.append("Signal")
    #         features.append(signals['Signal'][i])
    #         self.featuresNames.append("Position")
    #         features.append(signals['Position'][i])

    #         x.append(features)    

    #         if position == 1:
    #             y.append(1)
    #             buying += 1
    #         elif position == -1:
    #             y.append(0.5)
    #             selling += 1
    #         else:
    #             y.append(0)


    #         startIndex += block
    #         endIndex += block

    #     x = preprocessing.scale(x)
    #     # x = np.array(x)
    #     # scaler = StandardScaler()
    #     # x = scaler.fit_transform(x)
    #     x, y, prices = shuffle(x,y,prices)

    #     #return np.array(x), np.array(y), prices
    #     return np.array(x), np.array(y), prices


    # def createDataset2(self, cryptoData, isTraining, training_window = 180, labeling_window = 60, feature_window = 30):

    #     x = []
    #     y = []
    #     prices = []

    #     buying = 0
    #     selling = 0

    #     if isTraining:
    #         #btcData = read_csv('testData3/csv/trainingData.csv', names= Dataset.NUMERIC_COLUMNS)
    #         btcData = read_csv('testData3/csv/2021/01-2021.csv', names= Dataset.NUMERIC_COLUMNS)
    #     else:
    #         btcData = read_csv('testData3/csv/2021/02-2021.csv', names = Dataset.NUMERIC_COLUMNS)


    #     #Signals and moving averages

    #     shortWindow = 40
    #     longWindow = 100

    #     signals = DataFrame(index=btcData.index)
    #     signals['Signal'] = 0.0


    #     signals['short_mavg'] = btcData['Close'].rolling(window=shortWindow, min_periods=1, center=False).mean()

    #     signals['long_mavg'] = btcData['Close'].rolling(window=longWindow, min_periods=1, center=False).mean()

    #     signals['Position'] = 0.0

    #     #Features and Feature Names

    #     prevSignal = 0.0
    #     position = 0.0


    #     startIndex = 0
    #     endIndex = training_window - 1
    #     block = 1

    #     check = 0

    #     while endIndex < len(cryptoData) - labeling_window - block - 1:

    #         if check % 200 == 0:
    #             print("Checkpoint:",check)
    #         check += 1

    #         features = []
    #         self.featuresNames = []

    #         cur_low = cryptoData[endIndex][1]
    #         cur_high = cryptoData[endIndex][2]
    #         cur_open = cryptoData[endIndex][3]
    #         cur_close = cryptoData[endIndex][4]
    #         cur_volume = cryptoData[endIndex][5]



    #         for i in range(startIndex, endIndex):
    #             if i >= shortWindow:
                
    #         #Creating signal when short moving average crosses long moving average

    #                 if signals['short_mavg'][i] > signals['long_mavg'][i]:
    #                     signals['Signal'][i] = 1.0
    #                     position = 1.0 - prevSignal
    #                     signals['Position'][i] = position
    #                     prevSignal = 1.0
    #                 else:
    #                     signals['Signal'][i] = 0.0
    #                     position = 0.0 - prevSignal
    #                     signals['Position'][i] = position
    #                     prevSignal = 0.0
                
                        
    #             self.featuresNames.append("Signal")
    #             features.append(signals['Signal'][i])
    #             self.featuresNames.append("Position")
    #             features.append(signals['Position'][i])


    #         self.featuresNames.append("Low_Price")
    #         features.append(cur_low)
    #         self.featuresNames.append("High_Price")
    #         features.append(cur_high)
    #         self.featuresNames.append("Open_Price")
    #         features.append(cur_open)
    #         self.featuresNames.append("Volume")
    #         features.append(cur_volume)

    #         self.featuresNames.append("Close_Price")
    #         features.append(cur_close)
    #         prices.append(cur_close)


    #         x.append(features)    

    #         if signals['Position'][endIndex] == 1:
    #             y.append(1)
    #             buying += 1
    #         elif signals['Position'][endIndex] == -1:
    #             y.append(0.5)
    #             selling += 1
    #         else:
    #             y.append(0)

    #         startIndex += 1
    #         endIndex += 1


    #     print("Finished creating set of size "+ str(len(x)) + " " + str(len(y)))
    #     x = preprocessing.scale(x)
    #     x, y, prices = shuffle(x,y,prices)

    #     return np.array(x), np.array(y), prices


    def createDataset3(self, cryptoData, isTraining):
        x = []
        y = []
        prices = []

        buying = 0
        selling = 0

        if isTraining:
            btcData = read_csv('testData3/csv/trainingData.csv', names= Dataset.NUMERIC_COLUMNS)
            #btcData = read_csv('testData3/csv/2021/01-2021.csv', names= Dataset.NUMERIC_COLUMNS)
        else:
            btcData = read_csv('testData3/csv/2021/02-2021.csv', names = Dataset.NUMERIC_COLUMNS)


        #Signals and moving averages

        shortWindow = 40
        longWindow = 100

        signals = DataFrame(index=btcData.index)
        signals['Signal'] = 0.0


        signals['short_mavg'] = btcData['Close'].rolling(window=shortWindow, min_periods=1, center=False).mean()

        signals['long_mavg'] = btcData['Close'].rolling(window=longWindow, min_periods=1, center=False).mean()

        signals['Position'] = 0.0

        #Features and Feature Names

        prevSignal = 0.0
        position = 0.0

        for i in range(len(cryptoData)):

            features = []
            self.featuresNames = []

            cur_low = cryptoData[i][1]
            cur_high = cryptoData[i][2]
            cur_open = cryptoData[i][3]
            cur_close = cryptoData[i][4]
            cur_volume = cryptoData[i][5]

            if i >= shortWindow:
                
            #Creating signal when short moving average crosses long moving average

                if signals['short_mavg'][i] > signals['long_mavg'][i]:
                    signals['Signal'][i] = 1.0
                    position = 1.0 - prevSignal
                    signals['Position'][i] = position
                    prevSignal = 1.0
                else:
                    signals['Signal'][i] = 0.0
                    position = 0.0 - prevSignal
                    signals['Position'][i] = position
                    prevSignal = 0.0
                
                        
            self.featuresNames.append("Signal")
            features.append(signals['Signal'][i])
            self.featuresNames.append("Position")
            features.append(signals['Position'][i])


            self.featuresNames.append("Low_Price")
            features.append(cur_low)
            self.featuresNames.append("High_Price")
            features.append(cur_high)
            self.featuresNames.append("Open_Price")
            features.append(cur_open)
            self.featuresNames.append("Volume")
            features.append(cur_volume)

            self.featuresNames.append("Close_Price")
            features.append(cur_close)
            prices.append(cur_close)

            x.append(features)


            if position == 1:
                y.append(1)
                buying += 1
            elif position == -1:
                y.append(0.5)
                selling += 1
            else:
                y.append(0)


        print("Finished creating set of size "+ str(len(x)) + " " + str(len(y)))
        x = preprocessing.scale(x)
        x, y, prices = shuffle(x,y,prices)

        return np.array(x), np.array(y), prices



# dataset = Dataset()
# #dataset.createDataset()
# dataset.createDataset(dataset.loadCryptoData(TestingMonths), False)
