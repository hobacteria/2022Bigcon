# 소괄호 제거
import numpy as np
from geopy.geocoders import Nominatim
import pandas as pd
import re
from tqdm import tqdm
import requests
import json
import geopandas as gpd

path = 'C:/bigcon/'
import sys
sys.path.append(path + 'script')
from k_table import out, map_, df_res, df_act, df_res_act


def mk_yg(df): ## 앱실행 횟수 모음. 출퇴근 인구는 어떻게 처리할지?
    
    res = df.groupby(['adng_cd','adng_nm'])['count_cust'].sum()
    
    return res

youngin_grpsum = mk_yg(df_res) + mk_yg(df_act) + mk_yg(df_res_act)
youngin_grpsum = youngin_grpsum.to_frame().reset_index('adng_nm')







def kw_location(nm):
    
    global cnt
    
    url = 'https://dapi.kakao.com/v2/local/search/keyword.json' # 키워드 검색 url. 주소 검색 url이랑 다름.
    params = {'query': nm,
              'x':'127.177482330871' , ## 용인시 좌표를 중심으로 검색.
              'y':'37.241029979227',
              'category_group_code': 'CT1',
              'size' : 12}
    
    headers = {"Authorization": 'KakaoAK ' + appkey} # api 인증 key

    places = requests.get(url, params=params, headers=headers)

    xy = json.loads(places.text)
    tmp =[]
    
    for i in range(len(xy['documents'])):
        address_nm = dict(xy['documents'][i])['address_name']
        x = dict(xy['documents'][i])['x']
        y = dict(xy['documents'][i])['y']
        place_nm = dict(xy['documents'][i])['place_name']
        
        tmp.append([place_nm,address_nm,x,y])

    
    return pd.DataFrame(tmp)


def to_geo(df):

    def made(df,i):
        
        x = df['x'].iloc[i]
        y = df['y'].iloc[i]
        
        return f'POINT ({x} {y})'
    
    lst = []
    
    for i in range(len(df)):     
        lst.append(made(df,i))
    lst = gpd.GeoSeries.from_wkt(lst)
    
    
    
    df = gpd.GeoDataFrame(df,geometry = lst)
    
    return df

def _location(address,add_col,nm_col,df):
    
    global cnt
    url = 'https://dapi.kakao.com/v2/local/search/address.json'
    
    params = {'query': address,'analyze_type' : 'simillar', 'size' : 1}
    headers = {"Authorization": 'KakaoAK ' + appkey}

    places = requests.get(url, params=params, headers=headers)
    xy = json.loads(places.text)
    
    try:    
        address_nm = dict(xy['documents'][0])['address']['address_name']
        x = dict(xy['documents'][0])['x']
        y = dict(xy['documents'][0])['y']
        
    except:
        
        nm = df[df[add_col] == address][nm_col].head(1)
        
        return nm_location(nm)
    
    return [address_nm,x,y]

def nm_location(nm):
    
    global cnt
    
    url = 'https://dapi.kakao.com/v2/local/search/keyword.json' # 키워드 검색 url. 주소 검색 url이랑 다름.
    params = {'query': nm,
              'x':'127.177482330871' , ## 용인시 좌표를 중심으로 검색.
              'y':'37.241029979227',
              'size' : 1}
    
    headers = {"Authorization": 'KakaoAK ' + appkey} # api 인증 key

    places = requests.get(url, params=params, headers=headers)
    xy = json.loads(places.text)
    
    try:    
        address_nm = dict(xy['documents'][0])['address_name']
        x = dict(xy['documents'][0])['x']
        y = dict(xy['documents'][0])['y']
        
    except:
         cnt += 1
         if cnt % 50 == 0:
             print(f'no search result: {cnt}')
             
         return [nm,np.nan,np.nan]
    

    return [address_nm,x,y]



map_table = gpd.read_file(path + '/input/geo_dataset/HangJeongDong_ver20210401.geojson') #출처:https://github.com/vuski/admdongkor
map_table['adm_cd2'] = map_table['adm_cd2'].astype('float64')/100

map_table = map_table[map_table['sgg'].str.contains('^4146', na = False)]

map_table = map_table.set_index('adm_cd2').join(youngin_grpsum).reset_index()
map_ = map_table.copy()
map_table = map_table[['adng_nm','geometry']]


appkey = '63c767bfe1e2af023f72a87dca3ca6c9'
path = 'C:/bigcon/' # 경로 지정
movie = kw_location('용인시 영화관')
movie = movie[:10] #동탄 영화관 제외
movie.columns = ['nm','address','x','y']
movie = to_geo(movie)

movie_table = gpd.sjoin(map_table,movie,how = 'left').groupby(['adng_nm'])['x'].count()


mart = pd.read_csv(path + 'input/additional_dataset/경기도_용인시_대규모 점포 현황_20190409.csv',encoding = 'CP949')


mart[['주소', 'x', 'y']] = list(map(lambda x: _location(address = x,add_col = '소재지(사무실)',nm_col = '상호명', df = mart),mart['소재지(사무실)']))
mart = to_geo(mart)
mart_table = gpd.sjoin(map_table,movie,how = 'left').groupby(['adng_nm'])['x'].count()

tmp = movie_table + mart_table
tmp = tmp.fillna(0)


tmp.to_csv(path + 'output/data/checkpoint/ETC_places.csv',encoding = 'CP949')
