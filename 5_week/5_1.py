import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import r2_score

data = pd.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
#print(np.size(data,1))
train_data = data[[i for i in range(0,np.size(data,1)-1)]]
train_class = data[[np.size(data,1)-1]]
print(train_class)
kf = KFold(len(train_data), n_folds=5, shuffle=True, random_state=1)
#trange = np.linspace(1,50,num=50)
#rf = RandomForestRegressor(random_state=1, n_estimators=20)
#rf.fit(train_data,train_class['Rings'])

for tn in range(1,50):
    rf = RandomForestRegressor(random_state=1, n_estimators=tn)
    #rf.fit(train_data,train_class)
    cvs = cross_val_score(estimator=rf, X=train_data, y=train_class['Rings'], cv=kf, scoring='r2').mean()
    #print(cvs)
    if cvs>0.52:
        print(tn)
        break