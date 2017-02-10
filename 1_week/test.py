import pandas
import numpy as np
import heapq
from sklearn.tree import DecisionTreeClassifier
# from scipy.stats.stats import pearsonr
# from collections import Counter
data = pandas.read_csv('titanic.csv', index_col='PassengerId')
dataC = data[ ['Pclass','Fare','Age', 'Sex','Survived'] ]
dataCC = dataC.copy()
dataCC.Sex.loc[dataCC.Sex == 'male'] = 1
dataCC.Sex.loc[dataCC.Sex == 'female'] = 0
# for i in range(1,np.size(dataC,0)):
#     if(dataCC.loc[i,'Sex']=='male'):
#         dataCC.loc[i,'Sex']=1
#     else:
#         dataCC.loc[i,'Sex']=0


dataCC = dataCC[~np.isnan(dataCC.Age)]
# print(dataCC[:20])

truth = dataCC.Survived
features = dataCC[['Pclass','Fare','Age', 'Sex']]

model = DecisionTreeClassifier()
model.fit(features,truth)

importances = model.feature_importances_
print(importances)

def f(a,N):
    return np.argsort(a)[::-1][:N]

index = f(importances,2)
print(index)
headers = list(features.columns.values)
print(headers[index[1]], headers[index[0]])

open('2_1.txt','w').write(str(headers[index[1]])+" "+str(headers[index[0]]))

# print(heapq.nlargest(2,importances))


# data = pandas.read_csv('titanic.csv', index_col='PassengerId')
# # print(data[:10])
# M = np.sum(data['Sex']=='male')
# F = np.sum(data['Sex']=='female')
# print(M,F)
# Survived = np.sum(data['Survived'])/np.size(data['Survived'])*100
# print(Survived)
#
# FirstClass = np.sum(data['Pclass']==1)/np.size(data['Pclass'])*100
# print(FirstClass)
#
# MeanAge = np.nanmean(data['Age'])
# MedianAge = np.nanmedian(data['Age'])
# print(MeanAge, MedianAge)
# PC = pearsonr(data['SibSp'],data['Parch'])[0]
# print(PC)
#
# males = data[data.Sex =='male']
# females = data[data.Sex =='female']
#
# maleNames = list(males['Name'])
# femaleNames = list(females['Name'])
# noSecondName = femaleNames
# name = list()
# for i in range(0,np.size(femaleNames)):
#     noSecondName[i] = (femaleNames[i].split(","))[1]
#     noSecondName[i] = (femaleNames[i].split("."))[1]
#     bracket = noSecondName[i].find('(')
#     if bracket == -1:
#         space = noSecondName[i].find(' ', 1)
#         if space == -1:
#             name.append(noSecondName[i][1:])
#         else:
#             name.append(noSecondName[i][1:space])
#     else:
#         space = noSecondName[i].find(' ', bracket)
#         if space == -1:
#             name.append(noSecondName[i][bracket:])
#         else:
#             name.append(noSecondName[i][bracket+1:space])
#
#
# # cap_words = [word.upper() for word in name] #capitalizes all the words
# word_counts = (Counter(name)) #counts the number each time a word appears
#
# mostCommonFN = word_counts.most_common()[0][0]
# print(mostCommonFN)
#
# open('1.txt','w').write(str(M)+" "+str(F))
# open('2.txt','w').write(str(round(Survived,2)))
# open('3.txt','w').write(str(round(FirstClass,2)))
# open('4.txt','w').write(str(round(MeanAge,2))+" "+str(round(MedianAge,2)))
# open('5.txt','w').write(str(round(PC,2)))
# open('6.txt','w').write(mostCommonFN)
