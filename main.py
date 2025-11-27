import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
    This class implements a method to train an svm to classify handwritten numbers into even or odd
    @author: Balint Kovacs (677733)
"""


def loadData(path):
    """
    Loads the csv file into two matrices
    Args:
        path: (String) defines the path of the csv file
    Returns:
        d: Contains the labels of the training data 
        X: Contains the data on pixels
    """

    data = pd.read_csv(path)
    d = data.iloc[:, 0].to_numpy()
    X = data.iloc[:, 1:].to_numpy()
    return (d, X)


def image_data(X, i):
    """
        Transforms an entry of the pixel data into visualizable format
        Args:
            X: Matrix with the data
            i: Position of the desired observation
        Returns:
            A 28x28 matrix with data on pixel luminosity
    """

    pixelMatr = np.resize(X[i, :], (28, 28))
    return 255 - pixelMatr


def indicator(d, v):
    """
    Classifies a dataset based on certain critera
    Args:
        d: (numpy.ndarray) Array containing data to be classified
        v: A set containing the data in class 1

    Returns:
        A vector with entries that contain information about the classifications
    """

    return np.array([1 if di in v else -1 for di in d])

    """
    Extended approach:
        yData = []
        for i in range(len(d)):
            contains = False
            # A loop to check whether the element is contained by setting a boolean
            for j in range(len(v)):
                if d[i] == v[j]:
                    contains = True
            # Adding the value to yData[i]
            if contains:
                yData.append(1)
            else:
                yData.append(-1)
        return yData
    """


def func_loss(wbar, mu, X, y):
    """
    Computes the value of the loss function using the given parameters
    Args:
        wbar: input of function
        X: sample
        y: contains information about the classification of the observation
    Returns:
        Output of the function (scalar-valued)
    """

    N, p = X.shape
    w0 = wbar[0]
    w = wbar[1:]

    endVal = mu * (np.linalg.norm(w)) ** 2

    for i in range(N):
        plainDist = w0 + X[i, :] @ w
        subVal = max(0, 1 - y[i] * plainDist)
        endVal += subVal / N

    return endVal


def grad_loss(wbar, mu, X, y):
    """
    Computes the value of the gradient of the loss function
    Args:
        wbar: input of function
        X: sample
        y: contains information about the classification of the observation
    Returns:
        (numpy.ndarrray) Output of the function: value of the gradient
    """

    N, p = X.shape
    w0 = wbar[0]
    w = wbar[1:]

    r = 2 * mu * np.concatenate((np.array([0]), w))

    for i in range(N):
        plainDist = w0 + X[i, :] @ w
        if 1 - y[i] * plainDist >= 0:
            r += (-y[i] * np.concatenate((np.array([1]), X[i, :]))) / N

    return r


def svm(alpha, epsilon, mu, X, y):
    """
    A support vector machine that finds the optimal wbar value
    Args:
        alpha, epsilon, mu: scalar-valued, settings of the machine
    Returns:
        A matrix (numpy.ndarray) with the visited points. The last column is the optimal wbar value.
    """

    def f(wbar):
        """
            An inner function that simplifies the calling of the loss function
            Args:
                wbar: A vector for the input of the function
            Returns:
                The value of the func_loss using the arguments
        """
        return func_loss(wbar, mu, X, y)

    N, p = X.shape
    k = 0
    visits = np.zeros(p + 1).reshape((p + 1, 1))

    while k <= 1 or abs(f(visits[:, k]) - f(visits[:, max(0, k - 1)])) > epsilon:
        rk = grad_loss(visits[:, -1], mu, X, y)
        wNext = visits[:, -1] - alpha * rk
        visits = np.concatenate((visits, wNext.reshape((p + 1, 1))), axis=1)
        k += 1

    return visits[:, 1:]


def svm_prediction(wbar, X):
    """
    Classifies data based on an optimal vector.
    Args:
        wbar: The optimal vector calculated by the svm
        X: data matrix
    Returns:
        A vector containing information about the predicted classifications.
    """

    w0 = wbar[0]
    w = wbar[1:]
    N, p = X.shape

    yPred = []
    for i in range(N):
        plainDist = w0 + X[i, :] @ w
        if plainDist >= 0:
            yPred.append(1)
        else:
            yPred.append(-1)

    return yPred


def svm_accuracy(wbar, X, y):
    """
    Calculated the accuracy of the prediction made by the svm
    Args:
        wbar: The optimal wbar vector calculated by the svm
        X: data matrix
        y: the true classifications
    Returns:
        A fraction indicating how much of the classifications have been correct
    """

    w0 = wbar[0]
    w = wbar[1:]
    N, p = X.shape

    yHat = svm_prediction(wbar, X)

    correctCount = 0
    for i in range(N):
        if yHat[i] == y[i]:
            correctCount += 1

    return correctCount / N


def train_test_split(X, y, beta):
    """
    Splits the sample into a training and a testing sample
    Args:
        X: Sample matrix
        y: Classification data on the sample
        beta: Parameter for how much of the total data should go to training and how much should go to testing

    Returns:
        A tuple containing four elements: training data, training classifications, testing data, testing classifications
    """

    N, p = X.shape
    tresholdVal = int(N * beta)
    xTrain = X[:tresholdVal, :]
    yTrain = y[:tresholdVal]
    xTest = X[tresholdVal:, :]
    yTest = y[tresholdVal:]
    return (xTrain, yTrain, xTest, yTest)


def accuracies(X, y):
    """
    Executes the training of the svm based on the data set
    Args:
        X: Data matrix
        y: classification data
    Returns:
        A tuple containing the accuracy of the machine on the training data and outside the training data
    """

    xTrain, yTrain, xTest, yTest = train_test_split(X, y, 0.7)

    iterates = svm(10 ** -5, 10 ** -2, 10 ** -1, xTrain, yTrain)
    wbar = iterates[:, -1]

    inAcc = svm_accuracy(wbar, xTrain, yTrain)
    outAcc = svm_accuracy(wbar, xTest, yTest)

    return (inAcc, outAcc)



    # Used for testing:
def main():
    (dMain, xMain) = loadData('handwriting.csv')
    axes = plt.figure().add_axes((0,0,1,1))
    axes.imshow(image_data(xMain, 2), cmap='gray')
    plt.show()
    # dTest
    wbarTest = np.array([1,2,3,4])
    xTest = np.transpose(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    yTest = np.array([1, 0, 0])
    alpha = 10**-5
    epsilon = 10**-2
    mu = 10**-1
    evens = [0, 2, 4, 6, 8]
    yMain = np.array(indicator(dMain, evens))
    print(accuracies(X=xMain, y=yMain))

main()