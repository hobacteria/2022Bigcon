# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 21:55:56 2022

@author: ghrbs
"""

# 소괄호 제거
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import requests
import json

path = 'C:/bigcon/' # 경로 지정
 

ev_cg = pd.read_excel(path + 'input/additional_dataset/충전소 리스트.xlsx') # 데이터 불러오기
cols = ev_cg.iloc[1,:] # 데이터 전처리
ev_cg = ev_cg[2:]
ev_cg.columns = cols

ev_cg = ev_cg[ev_cg['시군구'].str.contains('^용인시',na = False)] # 용인시 데이터만 추출

ev_cg.loc[ev_cg[ev_cg['충전소'] == '동백금호어울림타운하우스A'].index,'충전소'] = '동백금호어울림타운하우스'
ev_cg.loc[ev_cg[ev_cg['충전소'] == '동백금호어울림타운하우스B'].index,'충전소'] = '동백금호어울림타운하우스'
ev_cg.loc[ev_cg[ev_cg['충전소'] == '서원마을금호베스트빌5단지 입주자대표회의'].index,'충전소'] = '서원마을금호베스트빌5단지'
ev_cg.loc[ev_cg[ev_cg['충전소'] == '경기 용인시 기흥구 트리플힐스로 7-12 (저압수용가)'].index,'주소'] = '경기 용인시 기흥구 트리플힐스로 7-12'
ev_cg.loc[ev_cg[ev_cg['충전소'] == '경기 용인시 기흥구 트리플힐스로 7-28 (저압수용가)'].index,'주소'] = '경기 용인시 기흥구 트리플힐스로 7-28'




pattern = r'\([^)]*\)' #괄호와 그 안의 문자를 제거해주는 정규표현
address = ev_cg['주소'] #충전소 데이터의 주소를 따로 저장
add_list = [] # 주소 처리 후 새로 담을 배열생성

for i in address: # 위의 패턴을 사용해 괄호 제거
    text = re.sub(pattern=pattern, repl='', string = i)
    add_list.append(text)
    
def sub_num(s): # 상세 주소에서 숫자와 '-'를 제외한 나머지 문자를 제거해주는 함수
    s = s[0:6]
    s = re.sub(r'[^0-9,^-]','',s)
    return(s)


for i in tqdm(range(len(add_list))):
    
    spl = 4 # 어디까지 멀쩡한 주소로 볼건지에 대한 기준. 보통은 4덩어리까지가 도, 시, 구, 로 구성으로 정상이고, 그 다음 상세 주소 덩어리가 하나 있음.
    
    a = add_list[i].split(' ') # 주소를 덩어리로 분리
    
    if len(a) < 3:
        pass
    
    if ('읍' in a[3]) or ('리' in a[3]) or ('면' in a[3]): # 주소에 읍면리가 들어가면 도/시/구/읍/로 가 된다. 따라서 멀쩡한 덩어리가 5개
        spl = 5
        
    a = a[0: spl + 1] # 상세 주소까지만 가져오고 나머지는 버리기
    
    if len(a) == spl:
        
        add_list[i] = " ".join(a[0:spl + 1]) # 상세주소가 없으면 그냥 그대로 가져오기
        
    elif len(a) == (spl + 1):
        
        a[-1] = sub_num(a[-1]) # 상세 주소에 대하여 위의 문자 제거 함수를 적용
        add_list[i] = " ".join(a[0:spl + 2]) # 다시 합치기
        
address_table = pd.DataFrame(address.copy())

address_table['수정주소'] = add_list
address_table.reset_index(drop = True,inplace = True)


#geo_df = list(map(geocoding,add_list))
 

## 카카오 api

tqdm.pandas()
cnt = 0
appkey = '63c767bfe1e2af023f72a87dca3ca6c9' # api키 그냥 넣어놨음. 일일 30만개 제한

def AK_location(address):
    global cnt # 글로벌 함수로 설정
    url = 'https://dapi.kakao.com/v2/local/search/address.json' ## api 주소
    
    params = {'query': address,'analyze_type' : 'simillar', 'size' : 1} # 요구 파라미터
    headers = {"Authorization": 'KakaoAK ' + appkey}

    places = requests.get(url, params=params, headers=headers)
    xy = json.loads(places.text)
    
    try:    
        nm = dict(xy['documents'][0])['address']['address_name']
        x = dict(xy['documents'][0])['x']
        y = dict(xy['documents'][0])['y']
        
    except:
         cnt += 1
         if cnt % 50 == 0:
             print(f'no search result: {cnt}') # 오류나면 카운트, 50개마다 출력
         return [address,np.nan,np.nan]
    

    return [nm,x,y]
xy_df = list(map(AK_location,add_list))
 

def emd_location(lst):
    url = 'https://dapi.kakao.com/v2/local/geo/coord2regioncode.json'
    
    x = lst[0]
    y = lst[1]
    
    
    params = {'x' : x,'y' : y}
    headers = {"Authorization": 'KakaoAK ' + appkey}

    places = requests.get(url, params=params, headers=headers)
    xy = json.loads(places.text)
    
    emd = dict(xy['documents'][0])['region_3depth_name']
    
   
    

    return emd

 

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
    
    emd = emd_location([x,y])
    return [emd,address_nm,x,y]



 
#카카오 api가 일일 트래픽 제한이 많기 떄문에 카카오 api로 먼저 시도하고, 안될 경우 tmap을 시도하는 코드로 수정

tqdm.pandas()
cnt = 0
appkey = '63c767bfe1e2af023f72a87dca3ca6c9'

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
        idx = address_table[address_table['수정주소'] == address][add_col].head(1)
        nm = ev_cg[ev_cg[add_col] == idx.iloc[0]][nm_col].head(1)
        
        return nm_location(nm)
    
    emd = emd_location([x,y])
    return [emd,address_nm,x,y]



 

_df = list(map(lambda x: _location(x, add_col = '주소', nm_col = '충전소',df = ev_cg),add_list))

df = pd.DataFrame(_df)

ev_cg[['emd','nm','x','y']] = _df


ev_cg.to_csv(path + 'output/data/checkpoint/전기차충전소_좌표.csv',encoding = 'CP949')
 
##

ap = pd.read_excel(path + 'input/additional_dataset/20220918_아파트정보목록.xlsx')
cnt = 0


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
    
    emd = emd_location([x,y])
    return [emd,address_nm,x,y]


   
_df = list(map(lambda x: _location(x, add_col = '도로명주소', nm_col = '단지명',df = ap),ap['도로명주소']))

## na열인 청명 센트레빌은 2019년 행정구역 경계 수정으로 인해 수원시에 편입됨
df = pd.DataFrame(_df)
df[df[1].isnull()]
ap[['emd','nm','x','y']]  = df
ap.dropna(inplace = True)

ap.to_csv(path + 'output/data/checkpoint/아파트정보목록 xy좌표.csv',encoding = 'CP949')


##
park_table = pd.read_excel(path + 'input/additional_dataset/parkigLotList (1).xlsx')
park_table[['x','y']] = park_table[['경도','위도']].astype(float)
type(park_table)
emdlst = []
for i in tqdm(range(len(park_table))):

    x = park_table['x'][i]
    y = park_table['y'][i]
    
    emdlst.append(emd_location([x,y]))
    
park_table['emd'] = emdlst
park_table.to_csv(path + 'output/data/checkpoint/용인도시공사_주차장 정보_20220621 xy좌표.csv', encoding = 'CP949')

