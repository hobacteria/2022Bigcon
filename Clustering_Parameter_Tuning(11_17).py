#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
모두 준수한 실루엣 점수를 보인다. 

random_state=0 #KMeans, MeanShift(bandwidth), GaussianMixture

KMeans
n = 6
Silhouette Coefficient: 0.5914
Davies Bouldin Index: 0.3922


AgglomerativeClustering
n = 5
Silhouette Coefficient: 0.5740
Davies Bouldin Index: 0.4102


AffinityPropagation
Silhouette Coefficient: 0.5823
Davies Bouldin Index: 0.4142

MeanShift
quantile=0.2
n_samples = 13
=> bandwidth
Silhouette Coefficient: 0.6012
Davies Bouldin Index: 0.3302

Birch
n=13
Silhouette Coefficient: 0.6229
Davies Bouldin Index: 0.3374

GaussianMixture
n =12
Silhouette Coefficient: 0.6247
Davies Bouldin Index: 0.3561
"""


# In[2]:


# 참고 ) https://tobigs.gitbook.io/tobigs/data-analysis/undefined-3/python-2-2


# ## 패키지 준비

# In[3]:


# data
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore") 

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

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

import copy


# ## 데이터 준비
# 결측치 확인, 상관계수 확인, PCA, MinMaxScaling 등

# In[4]:


path = 'C:/Users/KANGJIHUN/Desktop/빅자분/bigcon/'
df_pre = pd.read_csv(path + 'output\data\k_table.csv', encoding='CP949')


# In[5]:


df_pre.head(3)


# In[6]:


df_raw = pd.read_csv(path + 'output\data\k_table_final.csv', index_col=0)
df_raw.reset_index(drop = True, inplace = True)


# In[7]:


df_raw.head()


# In[8]:


df_raw['adng_nm'] = df_pre['adng_nm']


# In[9]:


df_raw.head(3)


# In[10]:


df_raw.to_csv(path + 'output/data/k_table_fixed.csv',encoding = 'cp949')


# In[11]:


df_corr = df_raw[:]


# In[12]:


df_corr.columns = ['#ofhouseholds', '#ofparkingAvailable', 'bpi', 'ev_traffic',
       'saturation', 'x_x', 'x_y', 'adng_nm']


# In[13]:


g = sns.heatmap(df_corr.corr(), annot=True, linewidths=.5)
bottom, top = g.get_ylim() # heatmap plot이 잘리는 것 방지하기 위한 방법
g.set_ylim(bottom+0.5, top-0.5)
plt.show()


# In[14]:


df = df_raw.iloc[:,:-1]


# In[15]:


df


# In[16]:


pca = PCA(n_components=2)
df = pca.fit_transform(df)
df.shape


# In[17]:


df = pd.DataFrame(df)


# ## K-means (1)
# elbow curve를 그려서 경사가 급격히 변하는 포인트를 확인

# In[18]:


distortions = []
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df) # df is array
    distortions.append(kmeans.inertia_)

fig = plt.figure(figsize=(10, 5))
plt.plot(range(2, 20), distortions)
plt.grid(True)
plt.title('Elbow curve')
plt.show()


# In[19]:


n_clusters = 5 # FIX ME 


# In[20]:


df = pd.DataFrame(df)


# In[21]:


n = [6] # 6
for i in n:
  km = KMeans(n_clusters=i, init='k-means++', random_state=0)
  cluster = km.fit(df) # df는 DataFrame
  cluster_id = pd.DataFrame(cluster.labels_)
  d1 = pd.concat([df, cluster_id], axis=1)


  d1.columns = [0, 1, "cluster"]
  sns.scatterplot(d1[0], d1[1], hue = d1['cluster'], legend="full")
  sns.scatterplot(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], label = 'Centroids')
  plt.title("KMeans Clustering")
  plt.legend()
  plt.show()


  # 실루엣 계수가 클수록, Davies Bouldin Index가 작을수록 좋다
  print('Silhouette Coefficient: {:.4f}'.format(metrics.silhouette_score(d1.iloc[:,:-1], d1['cluster'])))
  print('Davies Bouldin Index: {:.4f}'.format(metrics.davies_bouldin_score(d1.iloc[:,:-1], d1['cluster'])))


# In[22]:


km_labels = np.array(d1['cluster'])
km_X = np.array(d1.iloc[:,:-1])


# In[23]:


def get_silhouette_results(X, labels):
    def get_sum_distance(target_x, target_cluster):
        res = np.sum([np.linalg.norm(target_x-x) for x in target_cluster])
        return res
    
    '''
    각 데이터 포인트를 돌면서 a(i), b(i)를 계산
    그리고 s(i)를 계산한다.
    
    마지막으로 Silhouette(실루엣) Coefficient를 계산한다.
    '''
    uniq_labels = np.unique(labels)
    silhouette_val_list = []
    for i in range(len(labels)):
        target_data = X[i]
 
        ## calculate a(i)
        target_label = labels[i]
        target_cluster_data_idx = np.where(labels==target_label)[0]
        if len(target_cluster_data_idx) == 1:
            silhouette_val_list.append(0)
            continue
        else:
            target_cluster_data = X[target_cluster_data_idx]
            temp1 = get_sum_distance(target_data, target_cluster_data)
            a_i = temp1/(target_cluster_data.shape[0]-1)
 
        ## calculate b(i)
        b_i_list = []
        label_list = uniq_labels[np.unique(labels) != target_label]
        for ll in label_list:
            other_cluster_data_idx = np.where(labels==ll)[0]
            other_cluster_data = X[other_cluster_data_idx]
            temp2 = get_sum_distance(target_data, other_cluster_data)
            temp_b_i = temp2/other_cluster_data.shape[0]
            b_i_list.append(temp_b_i)
 
        b_i = min(b_i_list)
        s_i = (b_i-a_i)/max(a_i, b_i)
        silhouette_val_list.append(s_i)
 
    silhouette_coef_list = []
    for ul in uniq_labels:
        temp3 = np.mean([s for s, l in zip(silhouette_val_list, labels) if l == ul])
        silhouette_coef_list.append(temp3)
 
    silhouette_coef = max(silhouette_coef_list)
    return (silhouette_coef, np.array(silhouette_val_list))


# In[68]:


# 실루엣 계수 시각화 
def silhouette_visualize(df,labels):
    import seaborn as sns 
    import matplotlib.pyplot as plt
    import numpy as np 
    import seaborn as sns
    from matplotlib import font_manager, rc
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)

    ## 각 클러스터별로 Silhouette(실루엣) 값을 정렬한다.
    uniq_labels = np.unique(labels)
    sorted_cluster_svl = []
    rearr_labels = []
    label_nm = []
    for ul in uniq_labels:
        labels_idx = np.where(labels==ul)[0]
        target_svl = silhouette_val_list[labels_idx]
        sorted_cluster_svl += sorted(target_svl)
        rearr_labels += [ul]*len(target_svl)
        label_nm.extend(df_raw.loc[labels_idx,'adng_nm'])

    colors = sns.color_palette('Set3', len(uniq_labels))
    color_labels = [colors[i] for i in rearr_labels]
    
    fig = plt.figure(figsize=(6, 10))
    fig.set_facecolor('white')
    plt.vlines(metrics.silhouette_score(df.iloc[:,:-1], df['cluster']),0,35, color = 'red', linestyle = '--', linewidth = 2)
    plt.barh(range(len(sorted_cluster_svl)), sorted_cluster_svl, color=color_labels)
    plt.title("Average silthouette width: {:.4f}".format(metrics.silhouette_score(df.iloc[:,:-1], df['cluster'])))
    plt.yticks(np.arange(0,len(sorted_cluster_svl))[::-1],labels = label_nm)
    plt.ylabel('Data Index')
    plt.xlabel('Silhouette Value')
    plt.show()


# In[69]:


silhouette_coef, silhouette_val_list = get_silhouette_results(km_X, km_labels)
silhouette_visualize(d1,km_labels)


# ## Hierarchical agglomerative clustering
# 
# 특정 클러스터 개수보다 많아질 경우 Cluster 간의 거리가 급격히 줄어들면 자른다.

# In[70]:


linkage_array = ward(df)
dendrogram(linkage_array)
plt.xlabel("Sample Num")
plt.ylabel("Cluster Dist")

# 클러스터를 구분하는 커트라인을 표시
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [350, 350], '--', c='k')
ax.plot(bounds, [200, 200], '--', c='k')
ax.text(bounds[1], 350, ' 3 Clusters ', va='center', fontdict={'size': 10})
ax.text(bounds[1], 200, ' 5 Clusters ', va='center', fontdict={'size': 10})
plt.show()


# ## Agglomerative Clustering 

# In[71]:


n = [6] # 6
for i in n:
    agg = AgglomerativeClustering(n_clusters=i)
    cluster = agg.fit(df)
    cluster_id = pd.DataFrame(cluster.labels_)
    
    d4 = pd.DataFrame()
    d4 = pd.concat([df,cluster_id],axis=1)

    # 시각화 부분으로 PCA를 통해 변수를 2개로 줄이면 사용 가능

    d4.columns = [0, 1, "cluster"]
    
    sns.scatterplot(d4[0], d4[1], hue = d4['cluster'], legend="full")
    plt.title('Agglomerative with {} clusters'.format(i))
    plt.show()


    # 실루엣 계수가 클수록, Davies Bouldin Index가 작을수록 좋다
    print('Silhouette Coefficient: {:.4f}'.format(metrics.silhouette_score(d4.iloc[:,:-1], d4['cluster'])))
    print('Davies Bouldin Index: {:.4f}'.format(metrics.davies_bouldin_score(d4.iloc[:,:-1], d4['cluster'])))


# In[72]:


ag_labels = np.array(d4['cluster'])
ag_X = np.array(d4.iloc[:,:-1])


# In[73]:


silhouette_coef, silhouette_val_list = get_silhouette_results(ag_X, ag_labels)
silhouette_visualize(d4,ag_labels)


# ## Affinity Propagation (5)
# 모든 데이터가 특정한 기준에 따라 자신을 대표할 대표 데이터를 선택한다.
# 
# 만약 스스로가 자기 자신을 대표하게 되면 클러스터의 중심이 된다.
# 

# In[74]:


ap = AffinityPropagation() # 5
cluster = ap.fit(df)
cluster_id = pd.DataFrame(cluster.labels_)

d5 = pd.DataFrame()
d5 = pd.concat([df,cluster_id],axis=1)


d5.columns = [0, 1, "cluster"]

sns.scatterplot(d5[0], d5[1], hue = d5['cluster'], legend="full")
plt.title('Affinity Propagation {} clusters'.format(len(d5.cluster.unique())))
plt.show()


print('Silhouette Coefficient: {:.4f}'.format(metrics.silhouette_score(d5.iloc[:,:-1], d5['cluster'])))
print('Davies Bouldin Index: {:.4f}'.format(metrics.davies_bouldin_score(d5.iloc[:,:-1], d5['cluster'])))


# In[76]:


af_labels = np.array(d5['cluster'])
af_X = np.array(d5.iloc[:,:-1])

silhouette_coef, silhouette_val_list = get_silhouette_results(af_X, af_labels)
silhouette_visualize(d5,af_labels)


# In[77]:


len(silhouette_val_list)


# ### Mean Shift (6)
# - 평균 이동(Mean Shift)는 K-Means와 유사하게 중심을 군집의 중심으로 지속적으로 움직이며 군집화 진행
# - 평균 이동: 중심을 데이터가 모여 있는 밀도가 가장 높은 곳으로 이동한다.
# 
# 각 점들에 대해 데이터의 분포에서 mode를 찾아 이동하다보면 점들이 적당하게 모일 것이라는 것이라는 아이디어
# 
# bandwidth : 얼마나 관대하게 봐줄지를 설정하는 폭
# 
# ## 주로 사용하는 파라미터
# - bandwidth = n, bandwidth의 크기에 따른 군집화 영향도가 매우크다.
# - 최적의 bandwidth을 찾아야한다.
# 
# ### 참고
# - bandwidth의 적정값을 estimate_bandwidth함수로 출력 가능
# -  quantile = n,  데이터 개수의 일정 비율만큼 샘플링하면서 mean shift하게됨
# -  따라서, 데이터 개수가 엄청 많아질시 quantile값이 적으면 시간이 너무 오래걸림

# In[78]:


n = [13]    # 13
for i in n:
    bandwidth = estimate_bandwidth(df, quantile=0.2, n_samples=i, random_state=0)
    ms = MeanShift(bandwidth=bandwidth)
    cluster = ms.fit(df)
    cluster_id = pd.DataFrame(cluster.labels_)

    d6 = pd.DataFrame()
    d6 = pd.concat([df,cluster_id],axis=1)


    d6.columns = [0, 1, "cluster"]

    sns.scatterplot(d6[0], d6[1], hue = d6['cluster'], legend="full")
    plt.title('Mean Shift with {} samples'.format(i))
    plt.show()


    print('Silhouette Coefficient: {:.4f}'.format(metrics.silhouette_score(d6.iloc[:,:-1], d6['cluster'])))
    print('Davies Bouldin Index: {:.4f}'.format(metrics.davies_bouldin_score(d6.iloc[:,:-1], d6['cluster'])))


# In[79]:


ms_labels = np.array(d6['cluster'])
ms_X = np.array(d6.iloc[:,:-1])

silhouette_coef, silhouette_val_list = get_silhouette_results(ms_X, ms_labels)
silhouette_visualize(d6,ms_labels)


# In[80]:


len(ms_X)


# ## Birch (8)

# In[81]:


# n = [13]    # 13
# for i in n:
    
#     birch = Birch(n_clusters=i) # threshold 0.5, branching_factor 50
#     cluster = birch.fit(df)
#     cluster_id = pd.DataFrame(cluster.labels_) 

#     d8 = pd.DataFrame()
#     d8 = pd.concat([df,cluster_id],axis=1)


#     d8.columns = [0, 1, "cluster"]

#     sns.scatterplot(d8[0], d8[1], hue = d8['cluster'], legend="full")
#     plt.title('Birch with {} samples'.format(i))
#     plt.show()


#     print('Silhouette Coefficient: {:.4f}'.format(metrics.silhouette_score(d8.iloc[:,:-1], d8['cluster'])))
#     print('Davies Bouldin Index: {:.4f}'.format(metrics.davies_bouldin_score(d8.iloc[:,:-1], d8['cluster'])))


# In[82]:


# ms_labels = np.array(d6['cluster'])
# ms_X = np.array(d6.iloc[:,:-1])

# silhouette_coef, silhouette_val_list = get_silhouette_results(ms_X, ms_labels)
# silhouette_visualize(d6,ms_labels)


# ## Gaussian Mixture (11)

# In[83]:


n = [12]  # 12
for i in n:
    GM = GaussianMixture(n_components=i, random_state=0) # covariance_type{‘full’, ‘tied’, ‘diag’, ‘spherical’}, default=’full’
    cluster = GM.fit(df)
    cluster_id = pd.DataFrame(cluster.predict(df))

    d11 = pd.DataFrame()
    d11 = pd.concat([df,cluster_id],axis=1)

    
    d11.columns = [0, 1, "cluster"]

    sns.scatterplot(d11[0], d11[1], hue = d11['cluster'], legend="full")
    plt.title('Spectral Clustering with {} samples'.format(i))
    plt.show()
    

    print('Silhouette Coefficient: {:.4f}'.format(metrics.silhouette_score(d11.iloc[:,:-1], d11['cluster'])))
    print('Davies Bouldin Index: {:.4f}'.format(metrics.davies_bouldin_score(d11.iloc[:,:-1], d11['cluster'])))

# https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html (Model Selection 추가 bic, aic)
# https://rk1993.tistory.com/entry/AIC-BIC-Mallows-Cp-%EC%89%BD%EA%B2%8C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0 (bic, aic)
# AIC, BIC를 최소화한다는 것은 우도를 가장 크게 하는 동시에 변수 개수는 적은 최적의 모델을 찾는 것
# Clustering에선 변수 개수가 아니라 군집 개수일 듯


# In[84]:


gm_labels = np.array(d11['cluster'])
gm_X = np.array(d11.iloc[:,:-1])

silhouette_coef, silhouette_val_list = get_silhouette_results(gm_X, gm_labels)
silhouette_visualize(d11,gm_labels)


# In[ ]:




