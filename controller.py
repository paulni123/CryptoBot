



#Allow the user for the option to train model with current model and decide when to trade

#Also can create a prediction of future prices and when the model will trade based on those prices


from CryptoTrader import CryptoTrader
from Model import Model
from Dataset import Dataset
from Constants import *


def printInstructions():
    
    instructions = ['Type the word TRADE if looking to do so (must be properly set up with real credentials)',
    'Type FAKE for a simulation of the trading', 'Anything else will result in repeating of instructions']

    for string in instructions:
        print("*"+string +"*")

def tradeFunction():
    print("****WIP****")

    dataset = Dataset()

    data = dataset.loadCryptoData(TrainingMonths)
    trainX, trainY, _ = dataset.createDataset3(data, isTraining=True)

    data = dataset.loadCryptoData(TestingMonths)
    testX, testY, prices = dataset.createDataset3(data, isTraining=False)

    testModel = Model("CryptoTrader", trainX)
    testModel.train(trainX, trainY, batch_size=64, epochs=10)  

    cryptoTrader = CryptoTrader(model=testModel)
    cryptoTrader.simulate(testX, prices)  

def simulation():
    dataset = Dataset()

    data = dataset.loadCryptoData(TrainingMonths2)
    trainX, trainY, _ = dataset.createDataset3(data, isTraining=True)

    data = dataset.loadCryptoData(TestingMonths)
    testX, testY, prices = dataset.createDataset3(data, isTraining=False)

    testModel = Model("CryptoTrader", trainX)
    testModel.train(trainX, trainY, batch_size=64, epochs=10)  

    cryptoTrader = CryptoTrader(model=testModel)
    cryptoTrader.simulate(testX, prices)  
    

if __name__ == '__main__':


    print("Welcome to CryptoBot! \n\nHere are some of the available options:")
    printInstructions()
    
    choice = input()

    while choice != 'TRADE' and choice != 'FAKE':
        printInstructions()
        choice = input()
        

    if choice == 'TRADE':
        tradeFunction()
    else:
        simulation()




