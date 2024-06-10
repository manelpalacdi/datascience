from sklearn.preprocessing import StandardScaler
from printing import print_percentage
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# we want to find a linear function such as y = mx + b,
# updating m and b such that the difference of estimated y and 
# actual y (i.e. the loss/error) is minimized.
#
# then (m',b') = (m, b) - (dE/dm, dE/db) * lr, where lr is a 
# scalar factor multiplying the gradient descent of the error
# respect to m and b.

def normalize(val, min, max):
    return ((val - min)/(max - min))

def denormalize(val, min, max):
    return(val * (max - min) + min)

def gradient_descent(m, b, data, lr):
    dm = 0
    db = 0
    n = len(data)

    for i in range(n):
        x = data[i, 0]
        y = data[i, 1]
        dm += 1/n * x * ((m * x + b) - y)
        db += 1/n * ((m * x + b) - y)
    
    new_m = m - dm * lr
    new_b = b - db * lr

    return new_m, new_b


def main():


    # we check path of .csv
    path = "data.csv"
    if not (os.path.exists(path)):
        raise Exception("File or directory not found")
    
    # we get input for prediction
    try:
        x = float(input("\nWhat is the vehicle's mileage?\n"))
    except ValueError: 
        print("Wrong input, please type a number")

    data = pd.read_csv(path).to_numpy()
    n = len(data)

    # we normalize_data
    # the data so the linear regression computes smoother
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    ndata = (data - min_vals) / (max_vals - min_vals)
    
    # we split 80% to train and 20% to test
    train = ndata[:int(0.8 * n), :]
    test = ndata[int(0.8 * n):, :]

    # we set training reps and learning rate
    epochs = 1000
    lr = 0.1
    m = 0
    b = 0

    # the training begins
    for i in range(epochs):
        m, b = gradient_descent(m, b, train, lr)
        print_percentage(i, epochs)
    
    # normalize input, predict and invert normalize
    nx = normalize(x, min_vals[0], max_vals[0])
    est_y = m * nx + b
    est_y = denormalize(est_y, min_vals[1], max_vals[1])

    print(f"Prediction for {x:.2f}km is {est_y:.2f}$.\n")

    # we plot the results
    plt.scatter(data[:, 0], data[:, 1], c="k", s = 10)

    plt.plot([min_vals[0], max_vals[0]], 
             [denormalize(m * normalize(min_vals[0], min_vals[0], max_vals[0]) + b, min_vals[1], max_vals[1]), 
               denormalize(m * normalize(max_vals[0], min_vals[0], max_vals[0]) + b, min_vals[1], max_vals[1])], 
               c="r", 
               linestyle= "dashed")

    plt.plot(x, est_y, markersize=10, marker = "o", c="b")
    plt.annotate("predicted output", xy = (x, est_y), xytext=(x + 10000, est_y + 300), arrowprops=dict(arrowstyle="-"))

    plt.xlabel("Mileage")
    plt.ylabel("Price")

    plt.show()


if __name__ == "__main__":
    main()