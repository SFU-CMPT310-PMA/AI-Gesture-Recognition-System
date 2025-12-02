import tensorflow as tf
import numpy as np
import sys
from enum import Enum
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.layers import Normalization
from sklearn.model_selection import KFold
import random
import os

class HandSign(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    # UNKNOWN = 3       

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
    mapping = {"rock": 0, "paper": 1, "scissors": 2}
    y_norm = [str(lbl).strip().lower() for lbl in y]
    return np.array([mapping[lbl] for lbl in y_norm if lbl in mapping], dtype=np.int64)

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
def evaluateKFold(model, X, y, normalizer, numEpochs, batchSize):
    k: int = 10
    skf = KFold(n_splits=k, shuffle= True, random_state= 42)
    accuracies = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        yEncodedTrain = tf.keras.utils.to_categorical(toIntHandSign(y_train), len(HandSign)) 
        yEncodedTest = tf.keras.utils.to_categorical(toIntHandSign(y_test), len(HandSign)) 

        model = makeModel(inputDimension=X.shape[1], normalizer=normalizer)
        model.fit(X_train, yEncodedTrain, epochs=numEpochs, batch_size=batchSize, verbose=0)
        test_loss, test_accuracy = model.evaluate(X_test, yEncodedTest, verbose=0)
        accuracies.append(test_accuracy)

    print(f'Average Test Accuracy with KFold: {np.mean(accuracies) * 100:.2f}%')

def predictLabels(model, X):
    """
    Predict hand sign from features X.
    Returns: (label_str, confidence_float)
    """
    y_pred_distribution = model.predict(X, verbose=0)
    pred_conf = float(np.max(y_pred_distribution))
    pred_index = int(np.argmax(y_pred_distribution))
    pred_label = toLabelHandSigns(np.array([pred_index]))[0]
    return pred_label, pred_conf

def runModel(model, X_train, y_train, X_test, y_test, numEpochs, batchSize):    
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

def makeRandomModel(inputDimension, normalizer):
    """
    Create a model with random hyperparameters.
    """
    # Random hyperparameters
    hidden_units = random.choice(range(64))
    num_hidden_layers = random.choice(range(4))
    activation = random.choice(["relu", "tanh"])
    learning_rate = random.uniform(0.0005, 0.005)

    model = tf.keras.Sequential()
    model.add(normalizer)

    # Add hidden layers
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(hidden_units, activation=activation))

    # Output layer
    model.add(tf.keras.layers.Dense(3, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    # Return both model and the hyperparams for logging
    return model, {"hidden_units": hidden_units,
                   "num_hidden_layers": num_hidden_layers,
                   "activation": activation,
                   "learning_rate": learning_rate }

def train_multiple_models(X_train, y_train, X_test, y_test, normalizer, plotAccuracies):
    import os
    os.makedirs("models", exist_ok=True)

    results = []

    for i in range(10):
        print(f"\n==============================")
        print(f"Training Model #{i+1}")
        print("==============================")

        model, hparams = makeRandomModel(X_train.shape[1], normalizer)

        # Random training hyperparameters
        numEpochs = random.choice(range(20, 50))
        batchSize = random.choice(range(15))

        print("Hyperparameters:", hparams)
        print(f"Epochs: {numEpochs}, BatchSize: {batchSize}")

        history = model.fit(X_train, 
                            y_train,
                            epochs=numEpochs,
                            batch_size=batchSize,
                            validation_split=0.1,
                            verbose=0)

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {acc:.4f}")

        model_path = f"models/random_model_{i+1}.keras"
        model.save(model_path)
        print(f"Saved model to: {model_path}")

        # Store: accuracy, model hyperparams, training hyperparams, model_path
        results.append({"accuracy": acc,
                        "model_hparams": hparams,
                        "numEpochs": numEpochs,
                        "batchSize": batchSize,
                        "path": model_path})
        
        if plotAccuracies == 1:
            plotAccuracyAndLoss(history)

    # Sort by accuracy (best first)
    results.sort(key=lambda x: x["accuracy"], reverse=True)
    best = results[0]

    print("\n\n===== BEST MODEL =====")
    print("Accuracy:", best["accuracy"])
    print("Model Hyperparams:", best["model_hparams"])
    print("Epochs:", best["numEpochs"])
    print("BatchSize:", best["batchSize"])
    print("Saved:", best["path"])

    return results, best

def getDataset(path, include_unknown=False):
    data = pd.read_csv(path, sep=",", header=0)

    data = data.iloc[:, :64]

    if not include_unknown:
        data = data[data["label"].str.lower() != "unknown"]

    data = data.dropna()

    X = data.drop(columns=["label"]).to_numpy(dtype=np.float32)
    y_str = data["label"].str.lower().to_numpy()

    label_map = {"rock": 0, "paper": 1, "scissors": 2}
    y_int = np.array([label_map[lbl] for lbl in y_str])
    y_onehot = tf.keras.utils.to_categorical(y_int, num_classes=3)

    return y_str, y_onehot, X


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
    """
    Main controller to prepare dataset, train the model, and evaluate.
    """
    from sign_detection import prepareDataset, setUpNormalization, makeModel, runModel, comparePrediction, evaluateKFold

    numEpochs = 50
    batchSize = 32

    y_str, y_onehot, X = getDataset(path, include_unknown=False)

    X_train, y_train, X_test, y_test = prepareDataset(X, y_str)

    normalizer = setUpNormalization(X_train)
    model = makeModel(X.shape[1], normalizer)

    y_pred_labels = runModel(model, X_train, y_train, X_test, y_test, numEpochs, batchSize)

    comparePrediction(y_pred_labels, y_test)

    evaluateKFold(model, X, y_str, normalizer, numEpochs, batchSize)

def randomController(path, plotAccuracies):
    y_str, _, X = getDataset(path, include_unknown=False)

    X_train, y_train, X_test, y_test = prepareDataset(X, y_str)
    normalizer = setUpNormalization(X_train)

    _, best = train_multiple_models(X_train, y_train, X_test, y_test, normalizer, plotAccuracies)
    best_model_path = best["path"]

    best_model = tf.keras.models.load_model(best_model_path)
    

    evaluateKFold(best_model, X, y_str, normalizer, numEpochs=best["numEpochs"], batchSize=best["batchSize"])


def main():
    if len(sys.argv) > 1 and sys.argv[1].find("test") == 0:
        test()
    elif len(sys.argv) > 1 and sys.argv[1].find("random") == 0:
        if len(sys.argv) > 2 and sys.argv[2].find("plot") == 0:
            randomController('hand_gesture_dataset.csv', 1)
        else:
            randomController('hand_gesture_dataset.csv', 0)
    else:
        controller('hand_gesture_dataset.csv')

if __name__ == "__main__":
    main()
    print("Finished!")




################################## HOW TO USE THIS FILE ##################################
# Typical create 1 model: python3 sign_detection.py
# Create 10 random models: python3 sign_detection.py random
# Create 10 random models and plot each accuracy: python3 sign_detection.py random plot