import numpy as np
import pandas as pd
import math
from sklearn import metrics
train_data = pd.read_csv('data-logistic.csv', header = None)
y = train_data[0]
x1 = train_data[1]
x2 = train_data[2]

eps = 1e-5
max_iter = 1e+6
print(eps)
iter = 0
k_step = 0.1
Const = 1
0

w1 = 0
w2 = 0

while iter <= max_iter:
    w1_new = w1 + k_step* 1/len(y) * np.sum(y[i]*x1[i]*(1 - 1/(1 + math.exp(-y[i]*(w1*x1[i] + w2*x2[i])))) for i in range(len(y))) - k_step*Const*w1
    w2_new = w2 + k_step* 1/len(y) * np.sum(y[i]*x2[i]*(1 - 1/(1 + math.exp(-y[i]*(w1*x1[i] + w2*x2[i])))) for i in range(len(y))) - k_step*Const*w2
    if abs(w1 - w1_new) < eps and abs(w2-w2_new) < eps:
        w1 = w1_new
        w2 = w2_new
        print(w1, w2)
        break
    w1 = w1_new
    w2 = w2_new

    iter = iter + 1
y_reg = np.empty([len(y),1])
for i in range(len(y)):
    y_reg[i] = 1/(1 + math.exp(-w1*x1[i] - w2*x2[i]))

ras = metrics.roc_auc_score(y,y_reg)
print(ras)

open('3.txt','a').write(str(round(ras,3)))
