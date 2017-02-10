import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn import grid_search
from sklearn.feature_extraction.text import TfidfVectorizer
newsgroups = datasets.fetch_20newsgroups(subset='all',categories=['alt.atheism', 'sci.space'])
vectorizer = TfidfVectorizer()
data = vectorizer.fit_transform(newsgroups.data)
features = data
true = newsgroups.target
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(true.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(features, true)
C = gs.best_score_
est = gs.best_estimator_.C
print(C)
print(est)
model = SVC(C=est,kernel='linear', random_state=241)
model.fit(features, true)
coef0 = model.coef_.toarray()[0]
values = abs(coef0)
top10 = np.argsort(values)[-10:]
#coefabs = abs(model.coef_.data)
#print(coefabs)
#coefabssort = np.argsort(coefabs)[-10:]
feature_mapping = vectorizer.get_feature_names()
wr = []
for j in top10:
    print (j,feature_mapping[j])
    wr.append(feature_mapping[j])
    wr.sort(key=str.lower)
    print (wr)
#print(top10)

for i in top10:
    open('2.txt','a').write(str(feature_mapping[i])+' ')
