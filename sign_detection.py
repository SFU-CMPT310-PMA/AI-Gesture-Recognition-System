import tensorflow as tf
import numpy as np
import sys
from enum import Enum
import pandas as pd
from sklearn.model_selection import train_test_split

class HandSign(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    # UNKNOWN = 3       # Remove for now to test the model

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


def toIntHandSign(y):
    '''
    Argument:   Label y in numpy array
    Task:       Convert y labels from string ("rock", "paper", "scissor") to 
                Integer for the category match with HandSign
    '''
    try:
        return np.array([HandSign[label.upper()].value for label in y])
    except KeyError:
        print("Invalid Key. Return None...")
        return None

def toLabelHandSigns(y_pred):
    '''
    Argument:   Predicted y in integer HandSign class (numpy array)
    Task:       Convert the label in Integer to String for a better readability
    '''
    try:
        return np.array([HandSign(label).name for label in y_pred])
    except KeyError:
        print("Invalid Key. Return None...")
        return None

def comparePrediction(y_pred, y_test):
    '''
    Write result to csv file to compare the true label vs prediction
    '''
    data_compare = pd.DataFrame({'Predicted': y_pred, 'True': y_test, 'Correctness': y_pred == y_test})
    data_compare = data_compare.to_csv('compare_prediction.csv', sep=",")


def runModel(model, X, y):
    # Convert the string labels to integer corresponding to HandSign
    yHandSignValue = toIntHandSign(y)    

    # Convert the HandSign from Integer to Binary Class Matrix
    yEncoded = tf.keras.utils.to_categorical(yHandSignValue, len(HandSign))

    # Split the Dataset and Train the Model
    X_train, X_test, y_train, y_test = train_test_split(X, yEncoded, test_size=0.33, random_state=42)
    y_test = toLabelHandSigns(np.argmax(y_test, axis= 1))
    numEpochs: int = 50
    batchSize: int = 32
    model.fit(X_train, y_train, epochs=numEpochs, batch_size=batchSize)

    # Predict the Labels
    y_pred_distribution = model.predict(X_test)
    y_pred = np.argmax(y_pred_distribution, axis= 1)
    y_pred_label = toLabelHandSigns(y_pred)
    return y_pred_label, y_test


def makeModel(inputDimension: int):
    '''
    Create the model with 3 layers: 
    Layer 1 = 64 neurons with RELU activation
    Layer 2 = 32 neurons with RELU activation
    Layer 3 = 3 neurons with Softmax
    '''
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, input_dim=inputDimension, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def getDataset(path):
    '''
    1. Read dataset from the csv as a Pandas DataFrame
    2. Separate input vector (X_nparray) and labels (y)
    3. Convert X and y from DataFrame/Series to numpy array
    '''
    data: pd.DataFrame = pd.read_csv(path, sep=",", usecols=range(0, 64))
    y: pd.DataFrame = data["label"]
    X: pd.DataFrame = data.iloc[:, 1:]
    X_nparray = X.to_numpy()
    y_nparray = y.to_numpy()
    return y_nparray, X_nparray


def controller(path):
    y, X = getDataset(path)
    model = makeModel(X.shape[1])
    y_pred, y_test = runModel(model, X, y)
    comparePrediction(y_pred, y_test)


def main():
    if len(sys.argv) > 1 and sys.argv[1].find("test") == 0:
        test()
    else:
        controller('hand_gesture_dataset.csv')

if __name__ == "__main__":
    main()
    print("Ran!")