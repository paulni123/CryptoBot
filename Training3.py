from CoinbaseClient import CoinbaseClient
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import calendar

from sklearn.preprocessing import MinMaxScaler


def retrieveData():

    testClient = CoinbaseClient()
    #testClient.getCoinHistoricalData('BTC-USD',1625616000,1625702399, 300)

    #start_time = datetime.fromtimestamp(datetime(2021, 7, 1).timestamp())

    # with open('testData3/' + start_time.strftime('%m-%d-%y') +'.json', 'w') as f:
    #     f.write(json.dumps(testClient.getCoinHistoricalData('BTC-USD',start_time,end_time, 300)))

    start_time = datetime(2021,1,1)
    start_time_timestamp = start_time.timestamp()
    end_time = datetime(2021, 1, 2)
    end_time_timestamp = end_time.timestamp()

    numOfDays = calendar.monthrange(start_time.year, start_time.month)[1]

    tempList = []

    with open('testData3/' + start_time.strftime('%m-%Y') +'.json', 'w') as f:
        for _ in range(numOfDays):
            tempList += testClient.getCoinHistoricalData('BTC-USD',start_time,end_time, 300)[::-1]

            start_time_timestamp += 86400
            start_time = datetime.fromtimestamp(start_time_timestamp)
            end_time_timestamp += 86400
            end_time = datetime.fromtimestamp(end_time_timestamp)
        
        f.write(json.dumps(tempList))

    #Mon Jan 19 14:10:20 1970


def temp():
    testClient = CoinbaseClient()


    start_time = datetime(2021,7,15)
    start_time_timestamp = start_time.timestamp()
    end_time = datetime(2021, 7, 16)
    end_time_timestamp = end_time.timestamp()

    print(testClient.getCoinHistoricalData('BTC-USD',start_time,end_time, 300)[::-1])




trainingData = pd.read_csv('rawdata3.csv')
# print(trainingData)

trainingData = trainingData.iloc[:, 4].values  #Close values
# print(type(trainingData))

# print(trainingData)

scaler = MinMaxScaler()

trainingData = scaler.fit_transform(trainingData.reshape(-1,1))

retrieveData()
# temp()



