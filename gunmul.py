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


path = 'C:/bigcon/script'
import sys
sys.path.append(path)
from k_table import map_table,map_


path = 'C:/bigcon/' # 경로 지정
cnt = 0
appkey = '63c767bfe1e2af023f72a87dca3ca6c9' # api키 그냥 넣어놨음. 일일 30만개 제한
def many_location(nm, size,page):
    res = []
    global cnt
    
    
    url = 'https://dapi.kakao.com/v2/local/search/keyword.json' # 키워드 검색 url. 주소 검색 url이랑 다름.
    params = {'query': nm,
              'x':'127.177482330871' , ## 용인시 좌표를 중심으로 검색.
              'y':'37.241029979227',
              'page' : page,
              'size' : size}
    
    headers = {"Authorization": 'KakaoAK ' + appkey} # api 인증 key

    places = requests.get(url, params=params, headers=headers)
    xy = json.loads(places.text)
    
    for i in range(size):
        
        
        try:    
            place_nm = dict(xy['documents'][i])['place_name']
            address_nm = dict(xy['documents'][i])['address_name']
            x = dict(xy['documents'][i])['x']
            y = dict(xy['documents'][i])['y']
            
            r = [place_nm,address_nm,x,y]
            
        except:
            cnt += 1
            if cnt % 50 == 0:
                print(f'no search result: {cnt}')
                
            r = [nm,np.nan,np.nan,np.nan]
        
        
        res.append(r)
    
    res = pd.DataFrame(res)
    res.columns = ['place_nm','address_nm','x','y']
        
    #emd = emd_location([x,y])
    return res

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
    
def sjoin(table):
    table = to_geo(table)
    table = gpd.sjoin(map_table,table,how = 'left').groupby(['adng_nm'])['x'].count()
    return table

대학교_ = pd.DataFrame(many_location('대학교',15,1))
주유소_ = pd.DataFrame(many_location('주유소',15,1))
주유소_ = pd.concat([주유소_,pd.DataFrame(many_location('주유소',15,2))])
주유소_ = pd.concat([주유소_,pd.DataFrame(many_location('주유소',15,3))]).reset_index(drop = True)

종합병원_ = pd.DataFrame(many_location('종합병원',15,1))

대학교 = sjoin(대학교_)
종합병원 = sjoin(종합병원_)
주유소 = sjoin(주유소_)

gun_2  = 대학교 + 주유소

etc_place = pd.read_csv(path + 'output/data/checkpoint/ETC_places.csv',encoding = 'CP949',index_col='adng_nm')

gun_3 = etc_place['x'] + 종합병원

#gun_2.to_csv(path + 'output/data/checkpoint/건물2.csv')
#gun_3.to_csv(path + 'output/data/checkpoint/건물3.csv')

