import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import json
from CoinbaseClient import CoinbaseClient
from datetime import datetime
import calendar
import csv
import os


def retrieveData(year):

    testClient = CoinbaseClient()
    #testClient.getCoinHistoricalData('BTC-USD',1625616000,1625702399, 300)

    #start_time = datetime.fromtimestamp(datetime(2021, 7, 1).timestamp())

    # with open('testData3/' + start_time.strftime('%m-%d-%y') +'.json', 'w') as f:
    #     f.write(json.dumps(testClient.getCoinHistoricalData('BTC-USD',start_time,end_time, 300)))

    month = 1
    while month <= 12:

        start_time = datetime(year,month,1)
        start_time_timestamp = start_time.timestamp()
        end_time = datetime(year, month, 2)
        end_time_timestamp = end_time.timestamp()

        numOfDays = calendar.monthrange(start_time.year, start_time.month)[1]

        tempList = []

        title = start_time.strftime('%m-%Y')

        with open('testData3/json/'+ str(year) + '/' + title +'.json', 'w') as f:
            for _ in range(numOfDays):
                tempList += testClient.getCoinHistoricalData('BTC-USD',start_time,end_time, 300)[::-1]

                start_time_timestamp += 86400
                start_time = datetime.fromtimestamp(start_time_timestamp)
                end_time_timestamp += 86400
                end_time = datetime.fromtimestamp(end_time_timestamp)

            f.write(json.dumps(tempList))

        with open('testData3/json/'+ str(year) + '/' + title +'.json') as json_file:
            jsonData = json.load(json_file)

        trainingCSV = open('testData3/csv/'+ 'trainingData' +'.csv', 'a', newline='')
        csv_writer = csv.writer(trainingCSV)

        for data in jsonData:
            csv_writer.writerow(data)
        trainingCSV.close()        

        month += 1


    #Mon Jan 19 14:10:20 1970


def temp():
    # with open('testData3/json/2015/03-2015.json') as json_file:
    #     jsonData = json.load(json_file)

    # trainingCSV = open('testData3/csv/2015/03-2015.csv', 'w', newline='')
    # csv_writer = csv.writer(trainingCSV)

    # for data in jsonData:
    #     csv_writer.writerow(data)
    # trainingCSV.close()

    year = 2020
    while year < 2021:


        path = 'C:/Users/nipau/OneDrive/Desktop/CodingPrep/Projects/CryptoBot/testData3/json/' + str(year)
        # path2 = 'C:/Users/nipau/OneDrive/Desktop/CodingPrep/Projects/CryptoBot/testData3/csv/' + str(year)

        try:
            os.mkdir(path)
            # os.mkdir(path2)
        except OSError as error:
            pass

        retrieveData(year)

        year += 1


#retrieveData()
temp()
