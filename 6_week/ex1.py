from skimage.io import imread, imsave
from skimage import img_as_float
import pandas as pd
import numpy as np
import math
from sklearn import cluster
import pylab
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


image = imread('C:\\Users\\a.zakirov\\Documents\\Yandex-ML\\6_week\\parrots.jpg')
fimage = img_as_float(image)
w, h, d = image.shape
print(w,h,d)
image_r = np.reshape(fimage, (w*h, d))
image_r = pd.DataFrame(image_r,columns=['R','G','B'])
# print()

kmeans_model = cluster.KMeans(n_clusters=4, init = 'k-means++', random_state=241)
pixels_cluster = image_r.copy()
pixels_cluster['clusters'] = kmeans_model.fit_predict(image_r)
clusters_means = pixels_cluster.groupby('clusters').mean().values
pixels_mean = [clusters_means[c] for c in pixels_cluster['clusters'].values]
image_mean = np.reshape(pixels_mean, (w,h,d))
# imsave('parrots_' + str(8) + '.jpg', image_mean)

def image_clustering_by_mean(image,clusters):
    kmeans_model = cluster.KMeans(n_clusters=clusters, init = 'k-means++', random_state=241)
    pixels_cluster = image.copy()
    pixels_cluster['clusters'] = kmeans_model.fit_predict(image)
    clusters_means = pixels_cluster.groupby('clusters').mean().values
    pixels_mean = [clusters_means[c] for c in pixels_cluster['clusters'].values]
    clusters_median = pixels_cluster.groupby('clusters').median().values
    pixels_median = [clusters_median[c] for c in pixels_cluster['clusters'].values]
    return pixels_mean, pixels_median

# imsave('parrots_' + str(8) + '.jpg', image_mean)


def PSNR(data1, data2):
    maxi = np.mean(np.amax(image_r))
    mse = np.mean((data1 - data2)*(data1 - data2))
    psnr = 10 * math.log10(maxi*maxi/mse)
    return psnr

for i in range(1,20):
    pixels_mean, pixels_median = image_clustering_by_mean(image_r, i)
    image_mean = np.reshape(pixels_mean, (w,h,d))
    image_median = np.reshape(pixels_median, (w,h,d))
    psnr_mean = PSNR(fimage,image_mean)
    psnr_median = PSNR(fimage,image_median)
    if(psnr_mean>20 or psnr_median>20):
        print(psnr_mean, psnr_median, i)
        break
