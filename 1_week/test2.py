import pandas
import numpy as np
data = pandas.read_csv('titanic.csv', index_col='PassengerId')
dataC = data['Sex','Age']