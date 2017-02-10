import pandas as pd
import numpy as np
from sklearn.svm import SVC
data = pd.read_csv('svm-data.csv', header = None)
print(data[:10])
model = SVC(C=1000, random_state=241,kernel='linear')
features = data[[1, 2]]
true = data[[0]]
model.fit(features, true)
#p = model.predict(features)
s = model.support_
open('1.txt','w').write(str(s))