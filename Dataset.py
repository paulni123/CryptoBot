from pymongo import mongo_client
from CoinbaseClient import CoinbaseClient
import numpy as np
import json
import pymongo


# testClient = CoinbaseClient()


#     def storeHistoricalData(name):


# with open('testData/rawdata2.json', 'w') as f:
#     f.write(json.dumps(testClient.getCoinHistoricalData('BTC-USD',None,None,60)))
# print()


class Dataset:

    mongo_client = None
    mongo_database = None

    def __init__(self):
        print('created Dataset')
        self.attributeNames = []
        self.mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        self.mongo_database = self.mongo_client["HistoricalDataDB"]

    def storeHistoricalData(self, name, data):
        

        print('Storing data')
        #with open('testData/' + name + '.json', 'w') 

        col = self.mongo_database["DataSet"]
        #print(self.mongo_database.collection_names())


    # def createCollection(self, name):
    #     if

data = Dataset()

data.storeHistoricalData("name", "data")

print(data.mongo_database.collection_names())