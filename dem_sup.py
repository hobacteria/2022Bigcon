# 소괄호 제거
from cgitb import reset
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import requests
import json
import geopandas as gpd
from glob import glob
import haversine as hs
from haversine import Unit
##
path = 'C:/bigcon/script'
import sys
sys.path.append(path)

from k_table import map_table, map_,map_table
##
path = 'C:/bigcon/'
app_data = []
for i in sorted(glob(path + 'input/basic_dataset/*.csv')):
    app_data.append(pd.read_csv(i))
    

##

def to_geo(df):
    
    def made(df,i):
        
        x = df['x'].iloc[i]
        y = df['y'].iloc[i]
        
        return f'POINT ({x} {y})'
    
    lst = []
    
    for i in range(len(df)):     
        lst.append(made(df,i))
    lst = gpd.GeoSeries.from_wkt(lst)
    
    
    
    df = gpd.GeoDataFrame(df,geometry = lst,crs = 'EPSG:4326')
    
    return df


##
    
df_res = app_data[0] ## 주거자 데이터
df_act = app_data[1] ## 용인시에서 활동한 활동지 기준 데이터
df_res_act = app_data[2] ## 용인시 거주자가 활동한 지역 데이터

df_res_act = df_res_act.drop(['mega_nm','mega_cd'],axis = 1)

all_df = pd.concat([df_res_act,df_act,df_res], axis = 0)

def demand_place(dong,n = 10):
    tmp = all_df[all_df['adng_nm'] == dong]
    tmp = tmp.groupby(['cell_id','cell_xcrd','cell_ycrd'])['count_cust'].sum()
    tmp = tmp.sort_values(ascending = False).head(n)
    return tmp

# 성복동 죽전1동
sungbok_demand = demand_place('성복동',20).reset_index(['cell_xcrd','cell_ycrd'])
jukjun1_demand = demand_place('죽전1동',20).reset_index(['cell_xcrd','cell_ycrd'])


##

ap_table = pd.read_csv(path + 'output/data/checkpoint/아파트정보목록 xy좌표.csv', encoding = 'CP949')
park_table = pd.read_csv(path + 'output/data/checkpoint/용인도시공사_주차장 정보_20220621 xy좌표.csv', encoding = 'CP949')
ev_table = pd.read_csv(path + 'output/data/checkpoint/전기차충전소_좌표.csv', encoding = 'CP949')

ap_table = to_geo(ap_table)
park_table = to_geo(park_table)

ap_table['장소명'] = ap_table['단지명']
park_table['장소명'] = park_table['주차장명']

ap_table = gpd.sjoin(map_table,ap_table,how = 'left')
park_table = gpd.sjoin(map_table,park_table,how = 'left')


sungbok_sup = pd.concat([ap_table[ap_table['adng_nm'] == '성복동'][['x','y','geometry','장소명']],park_table[park_table['adng_nm'] == '성복동'][['x','y','geometry','장소명']]],axis = 0)
jukjun1_sup = pd.concat([ap_table[ap_table['adng_nm'] == '죽전1동'][['x','y','geometry','장소명']],park_table[park_table['adng_nm'] == '죽전1동'][['x','y','geometry','장소명']]],axis = 0)

sungbok_idx = sungbok_sup['장소명']
jukjun1_idx = jukjun1_sup['장소명']

sungbok_sup.to_csv(path + 'output/data/checkpoint/sungbok_sup.csv')
jukjun1_sup.to_csv(path + 'output/data/checkpoint/jukjun1_sup.csv')
sungbok_demand.to_csv(path + 'output/data/checkpoint/sungbok_demand.csv')
jukjun1_demand.to_csv(path + 'output/data/checkpoint/jukjun1_demand.csv')






##
def calc_dist(df_d,df_s):
    total_res = []

    for i in range(len(df_d)):
        res = []
        for j in range(len(df_s)):
            
            dem_x = df_d['cell_xcrd'].iloc[i]
            dem_y = df_d['cell_ycrd'].iloc[i]
            
            sup_x = df_s['x'].iloc[j]
            sup_y = df_s['y'].iloc[j]
            
            dist = hs.haversine((dem_y,dem_x), (sup_y ,sup_x), unit=Unit.METERS)
            res.append(dist)
        total_res.append(res)
    return pd.DataFrame(total_res)

sungbok = calc_dist(sungbok_demand,sungbok_sup)
jukjun1 = calc_dist(jukjun1_demand,jukjun1_sup)

sungbok.columns = sungbok_idx
jukjun1.columns = jukjun1_idx

sungbok['앱실행수'] = sungbok_demand['count_cust'].values
jukjun1['앱실행수'] = jukjun1_demand['count_cust'].values

sungbok.to_csv(path + 'output\data\checkpoint\sungbok_od_mat.csv',encoding= 'UTF-8')
jukjun1.to_csv(path + 'output\data\checkpoint\jukjun1_od_mat.csv',encoding= 'UTF-8')


