import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import coo_matrix, hstack
from sklearn.linear_model import Ridge

train_data = pd.read_csv('salary-train.csv')
test_data = pd.read_csv('salary-test-mini.csv')
train_data['FullDescription'] = train_data['FullDescription'].str.lower()
test_data['FullDescription'] = test_data['FullDescription'].str.lower()
#train_data['LocationNormalized'] = train_data['LocationNormalized'].str.lower()
train_data['FullDescription'] = train_data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
test_data['FullDescription'] = test_data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
#train_data['LocationNormalized'] = train_data['LocationNormalized'].replace('[^a-zA-Z0-9]', ' ', regex = True)
tvf = TfidfVectorizer(min_df=5)
# tvf2 = TfidfVectorizer()
data = tvf.fit_transform(train_data['FullDescription'])
#print(test_data[:10])
data_test = tvf.transform(test_data['FullDescription'])
train_data['LocationNormalized'].fillna('nan', inplace=True)
train_data['ContractTime'].fillna('nan', inplace=True)
test_data['LocationNormalized'].fillna('nan', inplace=True)
test_data['ContractTime'].fillna('nan', inplace=True)
div = DictVectorizer()
X_train_categ = div.fit_transform(train_data[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = div.transform(test_data[['LocationNormalized', 'ContractTime']].to_dict('records'))
X = hstack([data,X_train_categ])
X_test = hstack([data_test, X_test_categ])
# print(X_test)
# print(X)
y = train_data['SalaryNormalized']
clf = Ridge(alpha=1.0, random_state=241)
clf.fit(X,y)
y_test = clf.predict(X_test)
open('4_1.txt','w').write(str(np.round(y_test,2)))
print(y_test)
