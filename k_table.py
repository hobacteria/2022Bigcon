 
import pandas as pd
from glob import glob
import geopandas as gpd
 


path = 'C:/bigcon/' # 경로 지정

glob(path + '*좌표.csv')


ap_table = pd.read_csv(path + 'output/data/checkpoint/아파트정보목록 xy좌표.csv', encoding = 'CP949')
park_table = pd.read_excel(path + 'input/additional_dataset/parkigLotList (1).xlsx')
park_table[['x','y']] = park_table[['경도','위도']]
ev_table = pd.read_csv(path + 'output/data/checkpoint/전기차충전소_좌표.csv', encoding = 'CP949')
 

app_data = []

for i in sorted(glob(path + 'input/basic_dataset/*.csv')):
    app_data.append(pd.read_csv(i))
    
df_res = app_data[0] ## 주거자 데이터
df_act = app_data[1] ## 용인시에서 활동한 활동지 기준 데이터
df_res_act = app_data[2] ## 용인시 거주자가 활동한 지역 데이터

def mk_yg(df): ## 앱실행 횟수 모음. 출퇴근 인구는 어떻게 처리할지?
    
    res = df.groupby(['adng_cd','adng_nm'])['count_cust'].sum()
    
    return res

youngin_grpsum = mk_yg(df_res) + mk_yg(df_act) + mk_yg(df_res_act)
youngin_grpsum = youngin_grpsum.to_frame().reset_index('adng_nm')






 
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
    
 

ap_table = to_geo(ap_table)
park_table = to_geo(park_table)
ev_table = to_geo(ev_table)

# 전기차 교통량
ev_traffic = pd.read_excel(path + 'output/data/checkpoint/전기차 교통량.xlsx')

 

map_table = gpd.read_file(path + '/input/geo_dataset/HangJeongDong_ver20210401.geojson') #출처:https://github.com/vuski/admdongkor
map_table['adm_cd2'] = map_table['adm_cd2'].astype('float64')/100


map_table = map_table[map_table['sgg'].str.contains('^4146', na = False)]

map_table = map_table.set_index('adm_cd2').join(youngin_grpsum).reset_index()
map_ = map_table.copy()
map_table = map_table[['adng_nm','geometry']]



 
ap_table = gpd.sjoin(map_table,ap_table).groupby(['adng_nm'])['세대수'].sum()

park_table = gpd.sjoin(map_table,park_table).groupby(['adng_nm'])['주차구획수'].sum()


 
ev_table['완속충전기'] = list(map(lambda x: x.count('AC'), ev_table['충전기타입']))
ev_table = gpd.sjoin(map_table,ev_table).groupby(['adng_nm'])['완속충전기'].sum()


ev_traffic.set_index('index',inplace= True)
ev_traffic = ev_traffic[['전기차_교통량']]


tmp = pd.read_csv(path + 'output/data/checkpoint/ETC_places.csv',encoding = 'CP949',index_col = 'adng_nm')

area_table  = map_.area
area_table.index = map_['adng_nm']

 

out = pd.DataFrame()
out['기존충전소'] = ev_table
out['세대수'] = ap_table
out['주차가능수'] = park_table
out['주요건물수(영화관,대형점포)'] = tmp
out['ev_traffic'] = ev_traffic
out['면적'] = area_table
out
out = out.fillna(0)

out.to_csv(path + 'output/data/' + 'k_table.csv',encoding = 'CP949')

 

