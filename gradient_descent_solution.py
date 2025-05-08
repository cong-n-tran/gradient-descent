# Cong Tran 
# 1002046419

import numpy as np
import math

# f1(x, y) = x2 + 2y2 - 600x - 800y + xy + 50.

# foo(x,y) = sin(cos(x)+sin(2y))

# d(x, y)/dx = cos(cos(x) + sin(2y)) * (-sin(x))

# d(x, y)/dy = cos(cos(x) + sin(2y)) * (2cos(2y))

def foo_gradient(x: float, y: float) -> tuple: 
    dfdx = partial_derivative_x(x, y)
    dfdy = partial_derivative_y(x,y)
    return (dfdx, dfdy)


def partial_derivative_x(x: float, y: float) -> float: 
    return np.cos(np.cos(x) + np.sin(2 * y)) * -(np.sin(x))

def partial_derivative_y(x: float, y: float) -> float: 
    return np.cos(np.cos(x) + np.sin(2 * y)) * (2 * np.cos(2 * y))

def gradient_descent(function, gradient, x1: float, y1: float, eta: float, epsilon: float) -> tuple: 
    t = 1
    history = [(x1, y1)]
    while True: 
        xt, yt = history[-1]

        dfdx, dfdy = gradient(xt, yt)
        if math.sqrt(dfdx ** 2 + dfdy ** 2) < epsilon: 
            break

        new_xt, new_yt = np.array([xt, yt]) - eta * np.array([dfdx, dfdy])
        if function(new_xt, new_yt) > function(xt, yt): 
            eta = eta / 2
        else: 
            history.append((new_xt, new_yt))
            t +=1
    return (new_xt, new_yt, history)

