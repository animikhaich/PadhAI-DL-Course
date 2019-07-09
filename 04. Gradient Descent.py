import numpy as np

def sigmoid(w, b, x):
    return 1.0 / (1.0 + np.exp(-w*x + b))

def error(w, b, x, y):
    error = 0.0
    for x, y in zip(X, Y):
        fx = sigmoid(w, b, x)
        error += 0.5 * (fx - y) ** 2
    return error

def grad_b(w, b, x, y):
    fx = sigmoid(w, b, x)
    return (fx - y) * fx * (1 - fx)

def grad_w(w, b, x, y):
    fx = sigmoid(w, b, x)
    return (fx - y) * fx * (1 - fx) * x

def gradient_descent(w, b, lr, X, Y, epoch):
    for i in range(epoch):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        w -= dw * lr
        b -= db * lr
    return w, b

def predict(w, b, X, Y):
    fx = list()
    err = 0
    for x, y in zip (X, Y):
        fx.append(sigmoid(w, b, x))
        err += error(w, b, x, y)
    return np.round(fx), err

X = [0, 0, 1, 1]
Y = [0, 0, 0, 1]

w = 0
b = 0
lr = 0.001
epoch = 2000

w, b = gradient_descent(w, b, lr, X, Y, epoch)

print(predict(w, b, X, Y))