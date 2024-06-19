import pandas as pd
import numpy as np
import os
from logreg_train import logreg_train

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def print_prediction(Y, predictions_df):
    count = 0
    n = Y.shape[0]

    for i in range(n):
        if (Y[i] == predictions_df.iloc[i, 1]):
            count += 1
    correct_percentage = count / n * 100
    print(f"\n{correct_percentage:.2f}% of right guesses ({count}/{n})\n")
    print(predictions_df)

def logreg_predict(path):
    try:
        df = pd.read_csv(path)
        df.fillna(0, inplace=True)
        W = pd.read_csv("weighs.csv", header=None).values
        b = pd.read_csv("bias.csv", header=None).values
    except:
        print("Error reading .csv files")
    else:
        X = df.iloc[:, 6:].values
        Y = df.iloc[:, 1].values
        n = X.shape[0]

        predictions = []
        houses = {0 : "Gryffindor",
            1 : "Slytherin",
            2 : "Ravenclaw",
            3 : "Hufflepuff"}

        for i in range(n):
            Z = np.dot(W, X[i].T) + b.T
            Z = softmax(Z)
            prediction = houses[np.argmax(Z, axis=1)[0]]
            predictions.append([df.iloc[i, 2] + " " + df.iloc[i, 3], prediction])

        predictions_df = pd.DataFrame(predictions, columns=["Full name", "House"])
        
        print_prediction(Y, predictions_df)
        

if __name__ == "__main__":
    logreg_predict("datasets/dataset_train.csv")