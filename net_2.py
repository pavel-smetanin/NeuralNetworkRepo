from math import exp
from tkinter import Y
import openpyxl

#Для работы с векторами
def scalar(X, W):
    S = 0
    for i in range(len(X)):
        S += X[i] * W[i]
    return S

def sigmoid(x):
    return 1 / (1 + exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

##Класс нейрона 
class Neuron:
    def __init__(self):
        self.W = [0, 0, 0]
       # self.b = 0
    def get_W(self):
        return self.W
    def set_W(self, W):
        self.W = W
    def get_neuron_signal(self, input):
        S = scalar(input, self.W)
        return sigmoid(S)

class NeuralNetwork:
    def __init__(self):
        self.n1 = Neuron()
        self.n2 = Neuron()
        self.n3 = Neuron()
        self.n_ex = Neuron()
    def get_signal(self, input):
        s1 = self.n1.get_neuron_signal(input)
        s2 = self.n2.get_neuron_signal(input)
        s3 = self.n3.get_neuron_signal(input)
        s_exit = self.n_ex.get_neuron_signal([s1, s2, s3])
        return s_exit
    def set_W_in_n(self, matrix_w):
        self.n1.set_W(matrix_w[0])
        self.n2.set_W(matrix_w[1])
        self.n3.set_W(matrix_w[2])
    def set_W_in_n_ex(self, W):
        self.n_ex.set_W(W)
    def get_W_from_n(self):
        matrix_w = [[], [], []]
        matrix_w[0] = self.n1.get_W()
        matrix_w[1] = self.n2.get_W()
        matrix_w[2] = self.n3.get_W()
        return matrix_w
    def get_W_from_n_ex(self):
        return self.n_ex.get_W()

def F(W, Wex, X, Y):
    sum = 0
    for i in range(len(X)):
        sum += ((sigmoid(scalar(Wex, [sigmoid(scalar(X[i], W[0])), sigmoid(scalar(X[i], W[1])), sigmoid(scalar(X[i], W[2]))]))) - Y[i]) ** 2
    return sum

def gr_wex(Wex, W, X, Y, num):
    sum = 0
    for i in range(len(X)):
        list_hx = [sigmoid(scalar(X[i], W[0])), sigmoid(scalar(X[i], W[1])), sigmoid(scalar(X[i], W[2]))]
        sum += 2 * (Y[i] - sigmoid(scalar(list_hx, Wex))) * (-d_sigmoid(scalar(list_hx, Wex))) * sigmoid(scalar(X[i], W[num]))
    return sum

def gr_w(Wex, W, X, Y, n_list_W, n_W):
    sum = 0
    for i in range(len(X)):
        list_hx = [sigmoid(scalar(X[i], W[0])), sigmoid(scalar(X[i], W[1])), sigmoid(scalar(X[i], W[2]))]
        sum_wex = Wex[0] + Wex[1] + Wex[2]
        sum += 2 * ((Y[i] - sigmoid(scalar(list_hx, Wex))) * (-d_sigmoid(scalar(list_hx, Wex))) * sum_wex * d_sigmoid(scalar(X[i], W[n_list_W])) * X[i][n_W])
    return sum


list_in = []
list_out = []

book = openpyxl.load_workbook("neiron_set.xlsx")
sheet = book.active
for i in range(0, 150):
    list_in.append([sheet['B'+ str(i + 2)].value, sheet['C'+ str(i + 2)].value, sheet['D'+ str(i + 2)].value])
    list_out.append(sheet['E'+ str(i + 2)].value / 100)

network = NeuralNetwork()
L = 0.001
epochs = 100
matrix_Wn = network.get_W_from_n()
W_ex = network.get_W_from_n_ex()
print("Error before train: ", F(matrix_Wn, W_ex, list_in, list_out))
for ep in range(epochs):
    matrix_Wn_t = matrix_Wn.copy()
    W_ex_t = W_ex.copy()
    for i in range(len(W_ex)):
        W_ex[i] = W_ex[i] - L * gr_wex(W_ex, matrix_Wn, list_in, list_out, i)
    for i in range(len(matrix_Wn)):
        for j in range(len(matrix_Wn[i])):
            matrix_Wn[i][j] = matrix_Wn[i][j] - L*gr_w(W_ex_t, matrix_Wn_t, list_in, list_out, i, j)

print("Error after train: ", F(matrix_Wn, W_ex, list_in, list_out))
network.set_W_in_n(matrix_Wn)
network.set_W_in_n_ex(W_ex)

print("Signal with test: ", network.get_signal(list_in[91]), " True value: ", list_out[91])
print("Signal with test: ", network.get_signal(list_in[92]), " True value: ", list_out[92])
print("Signal with test: ", network.get_signal(list_in[100]), " True value: ", list_out[100])
