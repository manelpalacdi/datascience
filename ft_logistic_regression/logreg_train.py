import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from printing import print_percentage


def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def cross_entropy_loss(esty, y):
    return -np.sum(y * np.log(esty + 1e-9))  # Add a small epsilon to avoid log(0)

def stochastic_gradient_descent(X, Y, W, b, lr):
    n = X.shape[0]
    K = W.shape[0]
    F = W.shape[1]

    loss = 0

    # calculate est values for all classes and (x, y) train value pairs
    for i in range(n):
        Z = np.dot(X[i], W.T) + b.T # (z0, z1, z2, z3) shaped (1, 4) for each weight class
        sftZ = softmax(Z)
        loss += cross_entropy_loss(sftZ, Y[i])
        for k in range(K):
            for f in range(F):
                dWkf = (sftZ[k] - Y[i][k]) * X[i][f]
                W[k][f] -= lr * dWkf
            dbk = sftZ[k] - Y[i][k]
            b[k] -= lr * dbk
    return (loss / n)
        

def logreg_train(path: str):
    # dataframe from csv
    df = pd.read_csv(path)
    df.fillna(0, inplace=True) # avoid NaN

    # number of inputs:
    n = df.shape[0]
    # get the input features
    X = StandardScaler().fit_transform(df.iloc[:, 6:].values)
    # get the actual result
    Y = pd.get_dummies(df.iloc[:, 1]).values
    
    # Initialize parameters
    K = Y.shape[1]
    F = X.shape[1]
    W = np.zeros((K, F))
    b = np.zeros(K)

    loss = []

    epochs = 100
    for i in range(epochs):
        loss.append([i, stochastic_gradient_descent(X, Y, W, b, lr=0.1)])
        print_percentage(i, epochs)
    
    print(pd.DataFrame(loss, columns=["epoch", "loss"]))
    
    np.savetxt("weighs.csv", W, delimiter=",")
    np.savetxt("bias.csv", b, delimiter=",")

if __name__ == "__main__":
    logreg_train("datasets/dataset_train.csv")
