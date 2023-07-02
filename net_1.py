##Лабораторная работа №2
##Создание нейрона (4 входных и 1 выходной сигнал)
##Обучение нейросети из оного нейрона методом обратного распространения
from math import exp
from tkinter import Y

#Для работы с векторами
def scalar(X, W):
    S = 0
    for i in range(len(X)):
        S += X[i] * W[i]
    return S

def sigmoid(x):
    return 1 / (1 + exp(-x))

def grad_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

##Класс нейрона 
class Neuron:
    def __init__(self):
        self.W = [1, 2, 3, 4]
       # self.b = 0
    def get_W(self):
        return self.W
    def set_W(self, W):
        self.W = W
    def get_out_signal(self, input):
        S = scalar(input, self.W)
        return sigmoid(S)

def F(W, X, Y):
    sum = 0
    for i in range(len(X)):
        sum += (sigmoid(scalar(X[i], W)) - Y[i]) ** 2
    return sum

def gr_w1(W, X, Y):
    sum = 0
    for i in range(len(X)):
        sum += 2 * (sigmoid(scalar(X[i], W)) - Y[i]) * grad_sigmoid(scalar(X[i], W)) * X[i][0]
    return sum

def gr_w2(W, X, Y):
    sum = 0
    for i in range(len(X)):
        sum += 2 * (sigmoid(scalar(X[i], W)) - Y[i]) * grad_sigmoid(scalar(X[i], W)) * X[i][1]
    return sum

def gr_w3(W, X, Y):
    sum = 0
    for i in range(len(X)):
        sum += 2 * (sigmoid(scalar(X[i], W)) - Y[i]) * grad_sigmoid(scalar(X[i], W)) * X[i][2]
    return sum

def gr_w4(W, X, Y):
    sum = 0
    for i in range(len(X)):
        sum += 2 * (sigmoid(scalar(X[i], W)) - Y[i]) * grad_sigmoid(scalar(X[i], W)) * X[i][3]
    return sum

train_x = [[1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 1]]
train_y = [1, 0, 0, 0, 0]
test_x = [0, 1, 0, 1]
res_y = 0
n = Neuron()

L = 0.001
eps = 0.01
count = 0
Wt = n.get_W()
while True:
    Wp = Wt
    Wt[0] = Wt[0] - L*gr_w1(Wp, train_x, train_y)
    Wt[1] = Wt[1] - L*gr_w2(Wp, train_x, train_y)
    Wt[2] = Wt[2] - L*gr_w3(Wp, train_x, train_y)
    Wt[3] = Wt[3] - L*gr_w4(Wp, train_x, train_y)
    n.set_W(Wt)
    count += 1
    f = F(Wt, train_x, train_y)
    print("count = ", count, " W1 = ", Wt[0], " W2 = ", Wt[1], " W3 = ", Wt[2], " W4 = ", Wt[3], " F = ", f)
    if(abs(F(Wt, train_x, train_y) - F(Wp, train_x, train_y)) >= eps):
        break
print("count = ", count, " W1 = ", Wt[0], " W2 = ", Wt[1], " W3 = ", Wt[2], " W4 = ", Wt[3], " F = ", f)
print("Result with test signals: ", n.get_out_signal(test_x))
