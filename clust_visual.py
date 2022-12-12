# data
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore") 

# visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# model
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch

# grid search
from sklearn.model_selection import GridSearchCV

# evaluation
from sklearn.metrics.cluster import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import *

##visual
import seaborn as sns
from scipy.spatial import ConvexHull
from matplotlib import font_manager, rc

path = 'C:/bigcon/'
df_pre = pd.read_csv(path + 'output/data/k_table.csv', encoding='CP949')
df_raw = pd.read_csv(path + 'output/data/k_table_final.csv', index_col=0)
name = df_raw.index
df_raw['adng_nm'] = df_pre['adng_nm']
df = df_raw.iloc[:,:-1]

pca = PCA(n_components=2)
df = pca.fit_transform(df)
df = pd.DataFrame(df)
df.shape

# KMeans / 0.5914
# AgglomerativeClustering / 0.5740
# AffinityPropagation / 0.5823
# MeanShift / 0.6012
# Birch / 0.6229
# GaussianMixture / 0.6247


###########################################################################################################################
#kmean


X = np.array(df)
col = sns.color_palette('pastel')
ncluster = 6
cluster = KMeans(n_clusters=ncluster).fit(X)
y = cluster.labels_

centroids = cluster.cluster_centers_
fig, ax = plt.subplots(1, figsize=(7, 5))

def drawclusters(ax):
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
    for i in range(ncluster):
        points = X[y == i]
        ax.scatter(points[:, 0], points[:, 1], s=100, c=col[i], label=f'Cluster {i + 1}')
        
        for j in range(len(points)):
            plt.text(points[j, 0], points[j, 1], name[y==i][j],c=col[i])
            
        try:
            hull = ConvexHull(points)
            vert = np.append(hull.vertices, hull.vertices[0])  # close the polygon by appending the first point at the end
            ax.plot(points[vert, 0], points[vert, 1], '--', c=col[i])
            ax.fill(points[vert, 0], points[vert, 1], c=col[i], alpha=0.2)
        except:
            continue
        
    #ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', label='Centroids', marker='x')
    print('!')


drawclusters(ax)
ax.legend()
plt.title('k-means')
#plt.show()
plt.tight_layout()
plt.savefig(path + 'output/plot/k-meads.png')
plt.close()


###########################################################################################################################
# agg
agg = AgglomerativeClustering(6)
cluster = agg.fit(df)
#center = cluster.cluster_centers_

X = np.array(df)
col = sns.color_palette('pastel')
ncluster = 6
cluster = agg.fit(X)
y = cluster.labels_

centroids = cluster.cluster_centers_
fig, ax = plt.subplots(1, figsize=(7, 5))


drawclusters(ax)
ax.legend()
plt.title('Agglomerative Clustering')
#plt.show()
plt.tight_layout()
plt.savefig(path + 'output/plot/AgglomerativeClustering.png')
plt.close()

###########################################################################################################################
# aff
ncluster = 5
ap = AffinityPropagation() # 5
X = np.array(df)
cluster = ap.fit(X)
col = sns.color_palette('pastel')
y = cluster.labels_

centroids = cluster.cluster_centers_
fig, ax = plt.subplots(1, figsize=(7, 5))


drawclusters(ax)
ax.legend()
plt.title('Affinity Propagation')
#plt.show()
plt.tight_layout()
plt.savefig(path + 'output/plot/AffinityPropagation.png')
plt.close()


##mean shift
bandwidth = estimate_bandwidth(df, quantile=0.2, n_samples=13, random_state=0)
ms = MeanShift(bandwidth=bandwidth)
X = np.array(df)
cluster = ms.fit(X)
ncluster = 13
col = sns.color_palette('pastel')
y = cluster.labels_


#centroids = cluster.cluster_centers_
fig, ax = plt.subplots(1, figsize=(7, 5))

drawclusters(ax)
ax.legend()
plt.title('mean shift')
plt.tight_layout()
plt.savefig(path + 'output/plot/mean shift.png')
# plt.show()
plt.close()




## birch

birch = Birch(n_clusters=13) # threshold 0.5, branching_factor 50
X = np.array(df)
cluster = birch.fit(df)
ncluster = 13
col = sns.color_palette('pastel')

y = cluster.labels_

#centroids = cluster.cluster_centers_
fig, ax = plt.subplots(1, figsize=(7, 5))

drawclusters(ax)
ax.legend()
plt.title('Birch')
plt.tight_layout()
plt.savefig(path + 'output/plot/Birch.png')
# plt.show()
plt.close()



##GaussianMixture
X = np.array(df)
GM = GaussianMixture(n_components=12, random_state=0) # covariance_type{‘full’, ‘tied’, ‘diag’, ‘spherical’}, default=’full’
cluster = GM.fit(X)
ncluster = 12

col = np.array(sns.color_palette('pastel'))

col = col.hstack(col)

y = cluster.predict(df)

centroids = cluster.cluster_centers_
fig, ax = plt.subplots(1, figsize=(7, 5))

drawclusters(ax)
ax.legend()
plt.title('GaussianMixture')
plt.tight_layout()
plt.savefig(path + 'output/plot/GaussianMixture.png')
# plt.show()
plt.close()


