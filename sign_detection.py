import tensorflow as tf
import numpy as np
import sys
from enum import Enum
import pandas as pd

class HandSign(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    UNKNOWN = 3

def dataTranslator(inputData):
    # inputData is of type int[21][3]
    # each 2D vector will be passed in here to be flattened
    flattenedData = (np.array(inputData)).flatten()
    
    print(flattenedData)
    return flattenedData

def test():
    inputIdx = 1
    while len(sys.argv) > inputIdx:
        if sys.argv[inputIdx] == "testInputData":
            inputData = [
                [1.0, 2.0, 3.0],
                [4.0, 3.2, 0.5],
                [0.1, 2.8, 9.0] 
            ]
            translatedData = dataTranslator(inputData)
            inputIdx += 1
            print(translatedData)
        if len(sys.argv) > inputIdx and sys.argv[inputIdx] == "Dummy":
            print("Run another test")

def runModel(model, X, y):
    yEncoded = tf.keras.utils.to_categorical(y, len(HandSign))
    numEpochs: int = 50
    batchSize: int = 32
    model.fit(X, yEncoded, epochs=numEpochs, batch_size=batchSize)

def makeModel(inputDimension: int):
    model = tf.keras.Sequential()
    model.add(tf.keras.Dense(64, input_dim=inputDimension, activation='relu'))
    model.add(tf.keras.Dense(32, activation='relu'))
    model.add(tf.keras.Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def getDataset(path):
    data: pd.DataFrame = pd.read_csv(path, sep=",")
    y: pd.DataFrame = data["label"]
    X: pd.DataFrame = data.iloc[:, 1:]

    return y, X

def controller(path):
    y: pd.DataFrame; X: pd.DataFrame = getDataset(path)
    model = makeModel(X.shape[1])
    runModel(model, X, y)

def main():
    if len(sys.argv) > 1 and sys.argv[1].find("test") == 0:
        test()
    else:
        controller()

if __name__ == "__main__":
    # main()
    print("Ran!")