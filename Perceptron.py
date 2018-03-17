import numpy as np
import itertools
import math

# generating random initial weights [0-1)
def gen_w(n):
    print("generating random weights [0-1)")
    # w = np.random.uniform(0, 1, n)
    return np.tile(np.random.uniform(0, 1, n), (2 ** n, 1))


# generating initial weights as zeros
def gen_w_zeros(n):
    print("generating zero weights")
    return np.tile(np.zeros(n), (2 ** n, 1))


# generate binary vectors
def gen_x(n):
    print("generating binary vectors")
    x = [list(i) for i in (itertools.product([0, 1], repeat=n))]
    return np.asarray(x).astype(float)


# ask user perceptron input amount
def ask_vector_size():
    while True:
        try:
            size_choice = int(input("What size array would you like?\n"))
        except ValueError:
            print("Please input a number between 1 and 5")
            continue
        if 0 < size_choice < 5:
            break
        else:
            print("That is not between 1 and 5! Try again:")
    return size_choice


# ask user perceptron input weight type
def ask_initial_weight_type():
    while True:
        try:
            gen_choice = int(input("Do you want to: \n(1) Generate random weights [0;1)"
                                   "\n(2) Generate zero weights\n"))
        except ValueError:
            print("Please input a number")
            continue
        if 0 < gen_choice < 3:
            break
        else:
            print("That is not between 1 and 2! Try again:")
    return gen_choice


def calc_weighted_sum(w, x):
    return np.sum(w * x, axis=1)


def calc_step(a):
    y = np.zeros(a.size)
    i = 0
    for val in a:
        if val > 0:
            y[i] = 1
        else:
            y[i] = 0
        i += 1
    return y


def calc_sigmoid(a):
    y = np.zeros(a.size)
    i = 0
    for val in a:
        y[i] = 1 / (1 + math.exp(-val))
        i += 1
    return y


# let perceptron learn the weights
def perceptron_learn(w, x):
    # != optimal or iter<=maxiter
    a = calc_weighted_sum(w, x)
    i = 0
    while (e != 0):
        y = calc_step_func(a)
        i += 1
        w[i]
        e = ((y - t) ** 2).sum()


gen_w_dict = {1: gen_w, 2: gen_w_zeros}
vector_size = ask_vector_size()
weight_type = ask_initial_weight_type()
W = gen_w_dict[weight_type](vector_size)
X = gen_x(vector_size)
print(W)
print(X)

print(calc_sigmoid(calc_weighted_sum(W, X)))
eta = 1


# deltaW = -(eta * (deltaE / deltaW))
# for i in W:
#     W[i+1] = W[i] + deltaW

# W = gen_w(n)

