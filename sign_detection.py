import tensorflow as tf
import numpy as np
import sys
from enum import Enum
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Normalization
from sklearn.model_selection import KFold


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


##################################  CLEAN AND PREPARE THE DATASET     ##################################
def setUpNormalization(X_train):
    normalizer = Normalization(axis = -1)
    normalizer.adapt(X_train)
    return normalizer

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
    y_test = toLabelHandSigns(np.argmax(y_test, axis= 1))
    data_compare = pd.DataFrame({'Predicted': y_pred, 'True': y_test, 'Correctness': y_pred == y_test})
    data_compare = data_compare.to_csv('compare_prediction.csv', sep=",")

def prepareDataset(X, y):
    '''
    Split the Dataset into training, test and validation test
    '''
    # Convert the string labels to integer corresponding to HandSign
    yHandSignValue = toIntHandSign(y)    

    # Convert the HandSign from Integer to Binary Class Matrix
    yEncoded = tf.keras.utils.to_categorical(yHandSignValue, len(HandSign))

    # Split the Dataset and Train the Model
    X_train, X_test, y_train, y_test = train_test_split(X, yEncoded, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test




##################################  PREDICT AND EVALUATE MODEL FUNCTIONS     ################################## 
def evaluateKFold(model, X, y, normalizer):
    k: int = 10
    skf = KFold(n_splits=k, shuffle= True, random_state= 42)
    accuracies = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        yEncodedTrain = tf.keras.utils.to_categorical(toIntHandSign(y_train), len(HandSign)) 
        yEncodedTest = tf.keras.utils.to_categorical(toIntHandSign(y_test), len(HandSign)) 

        model = makeModel(inputDimension=X.shape[1], normalizer=normalizer)
        model.fit(X_train, yEncodedTrain, epochs=50, batch_size=32, verbose=0)
        test_loss, test_accuracy = model.evaluate(X_test, yEncodedTest, verbose=0)
        accuracies.append(test_accuracy)

    print(f'Average Test Accuracy with KFold: {np.mean(accuracies) * 100:.2f}%')



def runModel(model, X_train, y_train, X_test, y_test):
    numEpochs: int = 50
    batchSize: int = 32

    # Train the Model
    # Use 10% of training data for validation
    history = model.fit(X_train, y_train, epochs=numEpochs, batch_size=batchSize, validation_split=0.1, verbose=1)

    # Predict the Labels
    y_pred_distribution = model.predict(X_test)
    y_pred = np.argmax(y_pred_distribution, axis= 1)
    y_pred_label = toLabelHandSigns(y_pred)

    # Evaluate the Model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose= 0)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    # Visualize Accuracy and Loss
    plotAccuracyAndLoss(history)

    # Save model
    model.save("model/sign_model.keras")

    return y_pred_label


def makeModel(inputDimension: int, normalizer):
    '''
    Create the model with 3 layers: 
    Layer 1 = 64 neurons with RELU activation
    Layer 2 = 32 neurons with RELU activation
    Layer 3 = 3 neurons with Softmax
    '''
    model = tf.keras.Sequential()
    model.add(normalizer)
    model.add(tf.keras.layers.Dense(64, input_dim=inputDimension, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate = 0.005), metrics=['accuracy'])
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


##################################  PLOT ACCURACY AND LOSS     ##################################    
def plotAccuracyAndLoss(history):
    plt.figure(figsize=(12,5))

    #Plot Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label = 'Train Accuracy')
    plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label = 'Train Loss')
    plt.plot(history.history['val_loss'], label = 'Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


##################################  MAIN FUNCTION   ##################################
def controller(path):
    y, X = getDataset(path)
    X_train, y_train, X_test, y_test= prepareDataset(X, y)
    normalizer = setUpNormalization(X_train)
    model = makeModel(X.shape[1], normalizer)
    y_pred = runModel(model, X_train, y_train, X_test, y_test)
    comparePrediction(y_pred, y_test)
    evaluateKFold(model, X, y, normalizer)


def main():
    if len(sys.argv) > 1 and sys.argv[1].find("test") == 0:
        test()
    else:
        controller('hand_gesture_dataset.csv')

if __name__ == "__main__":
    main()
    print("Finished!")