import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
train_data = pd.read_csv('perceptron-train.csv', header = None)
test_data = pd.read_csv('perceptron-test.csv', header = None)
train_class = train_data[0]
test_class = test_data[0]
train_data = train_data[[1,2]]
test_data = test_data[[1,2]]
# print(train_class[:10])
pcp = Perceptron(random_state=241)
pcp.fit(train_data,train_class)
predict1 = pcp.predict(test_data)
acc1 = accuracy_score(test_class, predict1)
print(acc1)

scaler = StandardScaler()
train_data_sc = scaler.fit_transform(train_data)
test_data_sc = scaler.fit_transform(test_data)
pcp.fit(train_data_sc, train_class)
predict2 = pcp.predict(test_data_sc)
acc2 = accuracy_score(test_class,predict2)
print(acc2)
print(acc2-acc1)
open('3_1.txt','w').write(str(round(acc2-acc1,3)))


# задача 2
# boston = load_boston()
# boston.data = preprocessing.scale(boston.data)
# # print(boston.target)
# kf = KFold(len(boston.data), n_folds=5, shuffle=True, random_state=42)
# prange = np.linspace(1,10,num=200)
# max_from_mean = -100
# index = 0
# for p in prange:
#     knr = neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance', p=p, metric='minkowski')
#     cvs = cross_validation.cross_val_score(estimator=knr, X=boston.data, y=boston.target, cv=kf, scoring='mean_squared_error')
#     mean_cvs = cvs.mean()
#     print(mean_cvs)
#     if(mean_cvs > max_from_mean):
#         max_from_mean = mean_cvs
#         index = p
# open('2_1.txt','w').write(str(round(index,2)))


#  # задача 1
#

# # data = np.load("wine.data")
# data = pd.read_csv('wine.data', header = None)
#
#
# classes = data[0]
# features = data[[i for i in range(1,np.size(data,1))]]
# fearures_scales = preprocessing.scale(features)
# # print(features[:10])
#
#
# knn = neighbors.KNeighborsClassifier()
#
# def f(features,classes, kf):
#     max_from_mean = 0
#     for knn.n_neighbors in range(1,50):
#         cvs = cross_validation.cross_val_score(estimator=knn, X=features, y=classes, cv=kf, scoring='accuracy')
#         mean_cvs = cvs.mean()
#         # print(mean_cvs)
#         if(mean_cvs > max_from_mean):
#             max_from_mean = mean_cvs
#             index = knn.n_neighbors
#     return (max_from_mean, index)
#
# m_ns, k1 = f(features,classes,kf)
# m_s, k2 = f(fearures_scales,classes,kf)
#
# print(m_ns, k1, m_s, k2)
# open('2.txt','w').write(str(round(m_ns,2)))
# open('1.txt','w').write(str(k1))
# open('4.txt','w').write(str(round(m_s,2)))
# open('3.txt','w').write(str(k2))

