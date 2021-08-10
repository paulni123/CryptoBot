import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization


class Model:


    def __init__(self, modelName, trainX):
        self.model = self.createModel(trainX)
        self.modelName = modelName
        print("New model " + modelName + " created")



    def createModel(self,trainX):
        model = Sequential()

        xLength = len(trainX[0])

        for i in [(256,True), (128,True), (128,False)]:
            model.add(LSTM(i[0], input_shape = (1, xLength), return_sequences= i[1], activation= 'relu'))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(2, activation='softmax'))
        model.compile(loss= 'sparse_categorical_crossentropy', optimizer='adam')

        return model


    def train(self,trainX,trainY, batch_size, epochs):
        trainX = trainX.reshape(-1,1,len(trainX[0]))
        print("Training model - ",self.modelName)
        self.model.fit(trainX,trainY,batch_size = batch_size, epochs = epochs)

    
    def predict(self, sample):
        sample = sample.reshape(-1,1, len(sample[0]))
        prediction = np.array(tf.argmax(self.model.predict(sample),1))[0]
        return prediction





