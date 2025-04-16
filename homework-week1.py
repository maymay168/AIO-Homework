import math

# Exercise 1: Viet function thuc hien danh gia classification modal bang F1-Score

def exercise1(tp, fp, fn):
    if not isinstance(tp, int) or not isinstance(fp, int) or not isinstance(fn, int):
        if not isinstance(tp, int):
            print('tp must be int')
        if not isinstance(fp, int):
            print('fp must be int')
        if not isinstance(fn, int):
            print('fn must be int')
        return
    if tp <= 0 or fp <= 0 or fn <= 0:
        print('tp and fp and fn must be greater than zero')
        return
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1_score: {f1_score}')
     

# Exercise 2: Viet function mo phong theo 3 activation function

alpha = 0.01
def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True

def calc_activation_func(x, act_name):
    result = None
    if act_name == 'sigmoid':
        result = 1 / (1 + math.e**(-x))
    elif act_name == 'relu':
        if x > 0:
            result = x
        else:
            result = 0
    elif act_name == 'elu':
        if x > 0:
            result = x
        else:
            result = alpha * (math.e**x - 1)
    return result

def exercise2():
    x = input('Input x = ')
    if not is_number(x):
        print('x must be a number')
        return

    act_name = input('Input activation Function (sigmoid|relu|elu): ')
    x = float(x)
    result = calc_activation_func(x, act_name)
    if act_name is None:
        print(f'{act_name} is not supportted')
    else:
        print(f'{act_name}: f({x}) = {result}')
     

# Exercise 3: Viet function lua chon regression loss function de tinh loss
import random

def calc_ae(y, y_hat):
    result = abs(y - y_hat)
    return result

def calc_se(y, y_hat):
    result = (y - y_hat)**2
    return result

def exercise3():
    num_samples = input('Input number of samples (integer number) which are generated: ')
    if not num_samples.isnumeric():
        print("number of samples must be an integer number")
        return
    loss_name = input('Input loss name (MAE|MSE|RMSE): ')

    final_loss = 0
    num_samples = int(num_samples)
    for i in range(num_samples):
        pred_sample = random.uniform(0,10)
        target_sample = random.uniform(0,10)

        if loss_name == 'MAE':
            loss = calc_ae(pred_sample, target_sample)
        elif loss_name == 'MSE' or loss_name == 'RMSE':
            loss = calc_se(pred_sample, target_sample)
        final_loss += loss
        print(f'loss_name: {loss_name}, sample: {i}: pred: {pred_sample} target: {target_sample} loss: {loss}')

    final_loss /= num_samples
    if loss_name == 'RMSE':
        final_loss = math.sqrt(final_loss)
    print(f'{loss_name}: {final_loss}')

     

# Exercise 4: Viet 4 function de uoc luong cac ham sin(x), cos(x), sinh(x), cosh(x)

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def approx_cos(x, n):
    result = 0
    for i in range(n + 1):
        result += (-1)**i * x**(2 * i) / factorial(2 * i)
    return result

def approx_sin(x, n):
    result = 0
    for i in range(n + 1):
        result += (-1)**i * x**(2 * i + 1) / factorial(2 * i + 1)
    return result

def approx_sinh(x, n):
    result = 0
    for i in range(n + 1):
        result += (-1)**i * x**(2 * i + 1) / factorial(2 * i + 1)
    return result

def approx_cosh(x, n):
    result = 0
    for i in range(n + 1):
        result += (-1)**i * x**(2 * i) / factorial(2 * i)
    return result

     

# Exercise 5: Viet function thuc hien Mean Difference of nth Root Error:

def md_mre_single_sample(y, y_hat, n, p):
    result = (y**(1/n) - y_hat**(1/n))**p
    return result