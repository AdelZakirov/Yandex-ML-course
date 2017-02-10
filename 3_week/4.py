import numpy as np
import pandas as pd
import math
from sklearn import metrics
data = pd.read_csv('classification.csv', header = None)
#TP = np.sum((data[1:][1]))/np.sum((data[1:][0]))
true = data[1:][0]
pred = data[1:][1]

TP = np.sum((np.int_(true) + np.int_(pred)) == 2)
FP = np.sum((np.int_(true) - np.int_(pred)) == -1)
TN = np.sum((np.int_(true) + np.int_(pred)) == 0)
FN = np.sum((np.int_(true) - np.int_(pred)) == 1)
open('4_1.txt','w').write(str(TP)+' '+str(FP)+' '+str(FN)+' '+str(TN))

Accuracy = metrics.accuracy_score(true,pred)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_score = 2*TP/(2*TP+FP+FN)
open('4_2.txt','w').write(str(Accuracy)+' '+str(Precision)+' '+str(Recall)+' '+str(F1_score))
#Precision = metrics.precision_score(true, pred, average=None)
#Recall = metrics.recall_score(true, pred)
#F_score = metrics.f1_score(true, pred)

data2 = pd.read_csv('scores.csv', header = None)
true2 = np.int_(data2[1:][0])
score_logreg = np.float_(data2[1:][1])
score_svm = np.float_(data2[1:][2])
score_knn = np.float_(data2[1:][3])
score_tree = np.float_(data2[1:][4])
auc1 = metrics.roc_auc_score(true2, score_logreg)
auc2 = metrics.roc_auc_score(true2, score_svm)
auc3 = metrics.roc_auc_score(true2, score_knn)
auc4 = metrics.roc_auc_score(true2, score_tree)
#open('4_3.txt','w').write(str(auc1)+' '+str(auc2)+' '+str(auc3)+' '+str(auc4))

(precision1, recall1, thresholds1) = metrics.precision_recall_curve(true2, score_logreg)
(precision2, recall2, thresholds2) = metrics.precision_recall_curve(true2, score_svm)
(precision3, recall3, thresholds3) = metrics.precision_recall_curve(true2, score_knn)
(precision4, recall4, thresholds4) = metrics.precision_recall_curve(true2, score_tree)
recall1_2 = recall1[recall1>0.7]
recall2_2 = recall2[recall2>0.7]
recall3_2 = recall3[recall3>0.7]
recall4_2 = recall4[recall4>0.7]
precision1_2 = precision1[recall1>0.7]
precision2_2 = precision2[recall2>0.7]
precision3_2 = precision3[recall3>0.7]
precision4_2 = precision4[recall4>0.7]
max1 = np.max(precision1_2)
max2 = np.max(precision2_2)
max3 = np.max(precision3_2)
max4 = np.max(precision4_2)
print(max1, max2, max3, max4)
