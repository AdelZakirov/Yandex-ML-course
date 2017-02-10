import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

data = pd.read_csv('close_prices.csv')
train_data = data[[i for i in range(1,31)]]
# print(train_data[:10])
pca = PCA(n_components=10)
pca.fit(train_data)
train_data_pca = pca.transform(train_data)
# print(pca.explained_variance_ratio_)
# print(np.sum(pca.explained_variance_ratio_[[0,1,2,3]]))
first_component = train_data_pca[0:,0]
# print(first_component[:10])
# print(train_data_pca[:10])
dj = pd.read_csv('djia_index.csv')
data_dj = dj['^DJI']
# print(train_data[[np.argmax(pca.components_[0])]])
pc = np.corrcoef(first_component,data_dj)
print(pc)
open('4_2_2.txt','w').write(str(np.round(pc[0],2)))
# pc = pearsonr(first_component,data_dj)
# print(np.argmax(first_component))
# open('4_2.txt','w').write(str(np.round(y_test,2)))
