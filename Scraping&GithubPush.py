import os
import pickle
import locale
import copy
from urllib.request import urlopen
import json
from math import sqrt
import pandas as pd
import numpy as np
import dateparser
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px
from dash import Dash, dcc, Output, Input, html
import dash_bootstrap_components as dbc
from selenium import webdriver
from time import sleep
from bs4 import BeautifulSoup
from scrapy import selector
import requests
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.common.exceptions import NoSuchElementException
import time
import sys
import numpy as np
import openpyxl
from selenium.webdriver.firefox.options import Options
import locale
import io
import catboost
from catboost import CatBoost
locale.setlocale(locale.LC_ALL, 'tr_TR')
pd.set_option('display.max_columns', None)


with urlopen('https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr-cities-utf8.json') as response:
    harita = json.load(response)


df_old=pd.read_csv("https://raw.githubusercontent.com/alialtintass/Porsche/main/Porsche_old.csv", delimiter=",")
###################################################################################
from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
pd.set_option('display.max_columns', None)
options = Options()
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0) Gecko/20100101 Firefox/95.0"
options.set_preference("general.useragent.override", user_agent)


options.add_argument("--headless")
options.binary_location = '/Applications/Firefox.app/Contents/MacOS/firefox-bin'
options.set_preference("browser.download.folderList",2)
options.set_preference("browser.download.manager.showWhenStarting", False)
options.set_preference("browser.download.dir","/Data")
options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/octet-stream,application/vnd.ms-excel") 
driver = webdriver.Firefox(options=options, executable_path=GeckoDriverManager().install())
driver.get("https://www.sahibinden.com/porsche?pagingSize=50&sorting=date_desc")
driver.switch_to.window(driver.current_window_handle)
sleep(2)
#çerezleri kabul etme
driver.find_element("xpath",'//*[@id="onetrust-accept-btn-handler"]').click()
sleep(2)
#aramayı kaydet

driver.find_element("xpath",'/html/body/div[12]/div[3]').click()
sleep(2)
#tablo oluşumu
item_seri  = driver.find_elements("xpath",'//*[@id="searchResultsTable"]/tbody/tr[*]/td[2]')
item_model = driver.find_elements("xpath",'//*[@id="searchResultsTable"]/tbody/tr[*]/td[3]')
item_title = driver.find_elements("xpath",'//*[@id="searchResultsTable"]/tbody/tr[*]/td[4]')
item_year = driver.find_elements("xpath",'//*[@id="searchResultsTable"]/tbody/tr[*]/td[5]')
item_km = driver.find_elements("xpath",'//*[@id="searchResultsTable"]/tbody/tr[*]/td[6]')
item_color = driver.find_elements("xpath",'//*[@id="searchResultsTable"]/tbody/tr[*]/td[7]')
item_prices = driver.find_elements("xpath",'//*[@id="searchResultsTable"]/tbody/tr[*]/td[8]')
item_date = driver.find_elements("xpath",'//*[@id="searchResultsTable"]/tbody/tr[*]/td[9]')
item_loc = driver.find_elements("xpath",'//*[@id="searchResultsTable"]/tbody/tr[*]/td[10]')
#boş liste
seri_list=[]
model_list = []
titles_list = []
km_list =[]
color_list =[]
prices_list= []
date_list = []
loca_list = []
year_list =[]
# Loop over the item_titles and item_prices
for seri in item_seri:
    seri_list.append(seri.text)
for model in item_model:
    model_list.append(model.text)
for title in item_title:
    titles_list.append(title.text)
for year in item_year:
    year_list.append(year.text)
for km in item_km:
    km_list.append(km.text)
for color in item_color:
    color_list.append(color.text)
for prices in item_prices:
    prices_list.append(prices.text)
for dat in item_date:
    date_list.append(dat.text)
for locat in item_loc:
    loca_list.append(locat.text) 
df = pd.DataFrame(
    {'seri': seri_list,
    'model': model_list,
    'titles': titles_list,
     'year': year_list,
     'km' : km_list,
     'color': color_list,
     'prices': prices_list,
     'date': date_list,
     'location': loca_list
    })
while True:
    next_page_btn = driver.find_elements(by=By.LINK_TEXT, value="Sonraki")    
    try:
        next_page_btn[0].click()
        time.sleep(10)
        item_seri_new =  driver.find_elements("xpath",'//*[@id="searchResultsTable"]/tbody/tr[*]/td[2]')
        item_model_new = driver.find_elements("xpath",'//*[@id="searchResultsTable"]/tbody/tr[*]/td[3]')
        item_title_new = driver.find_elements("xpath",'//*[@id="searchResultsTable"]/tbody/tr[*]/td[4]')
        item_year_new =  driver.find_elements("xpath",'//*[@id="searchResultsTable"]/tbody/tr[*]/td[5]')
        item_km_new = driver.find_elements("xpath",'//*[@id="searchResultsTable"]/tbody/tr[*]/td[6]')
        item_color_new = driver.find_elements("xpath",'//*[@id="searchResultsTable"]/tbody/tr[*]/td[7]')
        item_prices_new = driver.find_elements("xpath",'//*[@id="searchResultsTable"]/tbody/tr[*]/td[8]')
        item_date_new = driver.find_elements("xpath",'//*[@id="searchResultsTable"]/tbody/tr[*]/td[9]')
        item_loc_new = driver.find_elements("xpath",'//*[@id="searchResultsTable"]/tbody/tr[*]/td[10]')
        #boş liste
        seri_list_new= []
        model_list_new = []
        titles_list_new = []
        km_list_new =[]
        color_list_new =[]
        prices_list_new= []
        date_list_new = []
        loca_list_new = []
        year_list_new =[]
# Loop over the item_titles and item_prices
        for seri in item_seri_new:
            seri_list_new.append(seri.text)
        for model in item_model_new:
            model_list_new.append(model.text)
        for title in item_title_new:
            titles_list_new.append(title.text)
        for year in item_year_new:
            year_list_new.append(year.text)
        for km in item_km_new:
            km_list_new.append(km.text)
        for color in item_color_new:
            color_list_new.append(color.text)
        for prices in item_prices_new:
            prices_list_new.append(prices.text)
        for dat in item_date_new:
            date_list_new.append(dat.text)
        for locat in item_loc_new:
            loca_list_new.append(locat.text) 
        df_new = pd.DataFrame(
            {'seri': seri_list_new,
            'model': model_list_new,
            'titles': titles_list_new,
            'year': year_list_new,
            'km' : km_list_new,
            'color': color_list_new,
            'prices': prices_list_new,
            'date': date_list_new,
            'location': loca_list_new
            })
        df = pd.concat([df, df_old])
        len(df)
        len(df_new)
    except IndexError:
        print("Last page reached")
        df.dropna(how='any',inplace=True)
        df.to_excel("Porsche_21022023.xlsx",index=False)
        break
    
df1=df.copy()
df=df_old.append(df)
len(df)

##################################################################################################
import github
from github import Github
from github import InputGitTreeElement
from datetime import datetime

file_list = [df]
file_names = ['Porsche_old.csv']
#Specify commit message
commit_message = 'Test Python'

#Create connection with GiHub
g = Github("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")

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir":
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))
list(df.columns)

df.to_csv("hede.csv", index=False)
with open('hede.csv', 'r', encoding="utf8") as file:
    content = file.read()

# Upload to github
git_prefix = 'Porsche/'
git_file = 'Porsche_old.csv'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.update_file(contents.path, "committing files",content , contents.sha, branch="main")
####################################################################################################################

df=df.drop_duplicates()
df = df.reset_index()
df['color'].replace('',np.nan,inplace=True)
df.dropna(subset=['color'], inplace=True)
df.reset_index(drop=True,inplace=True)
df[["Il", "Ilce"]] = df["location"].str.split(" ").apply(pd.Series)
df = df.drop('location', axis=1)

df['date'] = pd.to_datetime(df['date'],dayfirst=True, format='%d%B%Y')
df['date'] =  df['date'].dt.strftime('%d/%m/%Y')
df.date.head(5)
df['prices']=df['prices'].astype(str)
df.prices = df.prices.str.replace(' TL', '')
df.prices = df.prices.str.replace('.', '')
df['prices'] = pd.to_numeric(df['prices'])
df['km']=df['km'].astype(str)
df.km = df.km.str.replace(' TL', '')
df.km = df.km.str.replace('.', '')
df.km = df.km.str.replace(',', '')
df['km']=df['km'].astype(int)
df['year']=df['year'].astype(int)
choices = list()
conditions = list()
for item in harita['features']:
    conditions.append((df["Il"]==item['properties']['name']))
    choices.append(int(item['id']))

df['id'] = np.select(conditions, choices, default=0)
df.to_excel("ID_Control.xlsx")
df['prices'] = df['prices'].apply(pd.to_numeric, downcast='float', errors='coerce')
df['km_median'] =  df['km'].groupby([df['seri'], df['model'], df['year']]).transform('median')
df['km / median']= df.km/df['km_median']
df.drop('index', axis=1, inplace=True)
df.isnull().sum()
X = copy.deepcopy(df[['year', 'km', 'color', 'model', 'seri', 'km / median']])
y = copy.deepcopy(df[['prices']])
X = pd.get_dummies(data=X, columns=['color', 'year', 'model', 'seri'])#, drop_first=True)
#let us drop stuff we select in order to eliminate multicollinearity
X.drop('year_2022', axis=1,inplace=True)
X.drop('color_Bej', axis=1,inplace=True)
X.drop('model_Taycan', axis=1,inplace=True)
X.drop('seri_Taycan', axis=1,inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

data = copy.deepcopy(df[['prices','year', 'km', 'color', 'model', 'seri', 'km / median']])

from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.05)
yhat = iso.fit_predict(X_train)
mask = yhat != -1
X_train, y_train = X_train[mask], y_train[mask]
import catboost as cb
train_dataset = cb.Pool(X_train, y_train)
test_dataset = cb.Pool(X_test, y_test)
model = cb.CatBoostRegressor(loss_function='RMSE', eval_metric="RMSE",od_type="Iter")
grid = {'iterations' : [12000],
        'learning_rate': [0.008],
        'depth': [10],
        'l2_leaf_reg': [8],
        'early_stopping_rounds' : [3],
        'grow_policy':['Depthwise']
       }
model.grid_search(grid, train_dataset)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
pred = model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, pred)))
r2 = r2_score(y_test, pred)
print('Testing performance')
print('RMSE: {:.2f}'.format(rmse))
print('R2: {:.3f}'.format(r2))
#############################################################################################
df_grouped=df.groupby(['Il','id']).agg(avg_price=('prices','mean'),
                       max_price=('prices','max'),
                       min_price=('prices','min'),
                       data_counted=('Il','count')).reset_index()
df_grouped.set_index('id',inplace=True)

# df_grouped.columns
# df_grouped = pd.read_excel("test.xlsx")
# df_grouped.sort_index(ascending=True,inplace=True)
df_grouped['index1'] = df_grouped.index
df_grouped = df_grouped.reset_index(level=0)
df_grouped.sort_values('Il')
###########################################
file_list = [df_grouped]
file_names = ['df_grouped.csv']
#Specify commit message
commit_message = 'Test Python'

#Create connection with GiHub
g = Github("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")

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir":
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))
df_grouped.to_csv("hede.csv", index=False)
with open('hede.csv', 'r', encoding="utf8") as file:
    content = file.read()
# Upload to github
git_prefix = 'Porsche/'
git_file = 'df_grouped.csv'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.update_file(contents.path, "committing files",content , contents.sha, branch="main")
######################################################################################################

x_columns = list(X.columns)
unique_colors=list()
unique_years=list()
unique_models=list()
unique_seri=list()
for item in x_columns:
    if item[:5]=='color':
        unique_colors.append(item[6:])
    if item[:4]=='year':
        unique_years.append(int(item[5:]))
    if item[:5]=='model':
        unique_models.append(item[6:])
    if item[:4]=='seri':
        unique_seri.append(item[5:])
unique_years.append(2022)
unique_colors.append('Bej')
unique_models.append('Taycan')
unique_seri.append('Taycan')

data_colors=list(df.color.unique())
data_years=list(df.year.unique()).sort()
data_series=list(df.seri.unique())
df_trial=copy.deepcopy(df)
df_trial['date'] = pd.to_datetime(df_trial['date'], dayfirst=True)
series_weekly=df_trial.groupby([pd.Grouper(key='date', freq='W-MON')])['prices'].mean()
df_weekly=pd.DataFrame(data=series_weekly, columns=['prices'])
df_weekly.reset_index(inplace=True)

##########################################################################################
file_list = [df_trial]
file_names = ['df_trial.csv']
#Specify commit message
commit_message = 'Test Weekly'

#Create connection with GiHub
g = Github("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")

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir":
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))
df_trial.to_csv("df_trial.csv", index=False)
with open('df_trial.csv', 'r', encoding="utf8") as file:
    content = file.read()
# Upload to github
git_prefix = 'Porsche/'
git_file = 'df_trial.csv'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.update_file(contents.path, "committing files",content , contents.sha, branch="main")
###########################################################################################
file_list = [X_train]
file_names = ['X_train.csv']
#Specify commit message
commit_message = 'X_train'
#Create connection with GiHub
g = Github("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")
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir":
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))
X_train.to_csv("X_train.csv", index=False,encoding="utf8")
with open('X_train.csv', 'r', encoding="utf8") as file:
    content = file.read()
# Upload to github
git_prefix = 'Porsche/'
git_file = 'X_train.csv'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.update_file(contents.path, "committing files",content , contents.sha, branch="main")
##############################################################################
model.save_model('my_regressor.cbm')
model=model.load_model('my_regressor.cbm')
filename = 'my_regressor.cbm'
num_chunks = 20
with open(filename, 'rb') as f:
    file_size = os.path.getsize(filename)
    chunk_size = file_size // num_chunks
    for i in range(num_chunks):
        if i == num_chunks - 1:
            # last chunk may be larger than the others
            chunk_size = file_size - chunk_size * i
        chunk_filename = f"{os.path.splitext(filename)[0]}_part{i}.cbm"
        with open(chunk_filename, 'wb') as chunk_file:
            chunk = f.read(chunk_size)
            chunk_file.write(chunk)

#################################################################################
model_0 = b""

with open('my_regressor_part0.cbm', 'rb') as chunk_file:
    model_0 = chunk_file.read()
repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part0.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_0 = b""
with open('my_regressor_part0.cbm', 'rb') as chunk_file:
    model_0 = chunk_file.read()
model_0 = io.BytesIO(model_0)
repo.create_file("my_regressor_part0.cbm", "commit message",model_0.getvalue())

################
model_1 = b""
with open('my_regressor_part1.cbm', 'rb') as chunk_file:
    model_1 = chunk_file.read()

all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part1.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_1 = b""
with open('my_regressor_part1.cbm', 'rb') as chunk_file:
    model_1 = chunk_file.read()
model_1 = io.BytesIO(model_1)
repo.create_file("my_regressor_part1.cbm", "commit message",model_1.getvalue())

################
model_2 = b""
with open('my_regressor_part2.cbm', 'rb') as chunk_file:
    model_2 = chunk_file.read()

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part2.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_2 = b""
with open('my_regressor_part2.cbm', 'rb') as chunk_file:
    model_2 = chunk_file.read()
model_2 = io.BytesIO(model_2)
repo.create_file("my_regressor_part2.cbm", "commit message",model_2.getvalue())

################
model_3 = b""
with open('my_regressor_part3.cbm', 'rb') as chunk_file:
    model_3 = chunk_file.read()

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part3.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_3 = b""
with open('my_regressor_part3.cbm', 'rb') as chunk_file:
    model_3 = chunk_file.read()
model_3 = io.BytesIO(model_3)
repo.create_file("my_regressor_part3.cbm", "commit message",model_3.getvalue())

################
model_4 = b""
with open('my_regressor_part4.cbm', 'rb') as chunk_file:
    model_4 = chunk_file.read()

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part4.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_4 = b""
with open('my_regressor_part4.cbm', 'rb') as chunk_file:
    model_4 = chunk_file.read()
model_4 = io.BytesIO(model_4)
repo.create_file("my_regressor_part4.cbm", "commit message",model_4.getvalue())

################
model_5 = b""
with open('my_regressor_part5.cbm', 'rb') as chunk_file:
    model_5 = chunk_file.read()

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part5.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_5 = b""
with open('my_regressor_part5.cbm', 'rb') as chunk_file:
    model_5 = chunk_file.read()
model_5 = io.BytesIO(model_5)
repo.create_file("my_regressor_part5.cbm", "commit message",model_5.getvalue())
################
model_6 = b""
with open('my_regressor_part6.cbm', 'rb') as chunk_file:
    model_6 = chunk_file.read()

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part6.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_6 = b""
with open('my_regressor_part6.cbm', 'rb') as chunk_file:
    model_6 = chunk_file.read()
model_6 = io.BytesIO(model_6)
repo.create_file("my_regressor_part6.cbm", "commit message",model_6.getvalue())

################
model_7 = b""
with open('my_regressor_part7.cbm', 'rb') as chunk_file:
    model_7 = chunk_file.read()

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part7.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_7 = b""
with open('my_regressor_part7.cbm', 'rb') as chunk_file:
    model_7 = chunk_file.read()
model_7 = io.BytesIO(model_7)
repo.create_file("my_regressor_part7.cbm", "commit message",model_7.getvalue())
################
model_8 = b""
with open('my_regressor_part8.cbm', 'rb') as chunk_file:
    model_8 = chunk_file.read()

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part8.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_8 = b""
with open('my_regressor_part8.cbm', 'rb') as chunk_file:
    model_8 = chunk_file.read()
model_8 = io.BytesIO(model_8)
repo.create_file("my_regressor_part8.cbm", "commit message",model_8.getvalue())

################
model_9 = b""
with open('my_regressor_part9.cbm', 'rb') as chunk_file:
    model_9 = chunk_file.read()

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part9.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_9 = b""
with open('my_regressor_part9.cbm', 'rb') as chunk_file:
    model_9 = chunk_file.read()
model_9 = io.BytesIO(model_9)
repo.create_file("my_regressor_part9.cbm", "commit message",model_9.getvalue())

################
model_10 = b""
with open('my_regressor_part10.cbm', 'rb') as chunk_file:
    model_10 = chunk_file.read()

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part10.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_10 = b""
with open('my_regressor_part10.cbm', 'rb') as chunk_file:
    model_10 = chunk_file.read()
model_10 = io.BytesIO(model_10)
repo.create_file("my_regressor_part10.cbm", "commit message",model_10.getvalue())

################
model_11 = b""
with open('my_regressor_part11.cbm', 'rb') as chunk_file:
    model_11 = chunk_file.read()

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part11.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_11 = b""
with open('my_regressor_part11.cbm', 'rb') as chunk_file:
    model_11 = chunk_file.read()
model_11 = io.BytesIO(model_11)
repo.create_file("my_regressor_part11.cbm", "commit message",model_11.getvalue())

################
model_12 = b""
with open('my_regressor_part12.cbm', 'rb') as chunk_file:
    model_12 = chunk_file.read()

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part12.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_12 = b""
with open('my_regressor_part12.cbm', 'rb') as chunk_file:
    model_12 = chunk_file.read()
model_12 = io.BytesIO(model_12)
repo.create_file("my_regressor_part12.cbm", "commit message",model_12.getvalue())

################
model_13 = b""
with open('my_regressor_part13.cbm', 'rb') as chunk_file:
    model_13 = chunk_file.read()

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part13.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_13 = b""
with open('my_regressor_part13.cbm', 'rb') as chunk_file:
    model_13 = chunk_file.read()
model_13 = io.BytesIO(model_13)
repo.create_file("my_regressor_part13.cbm", "commit message",model_13.getvalue())

################
model_14 = b""
with open('my_regressor_part14.cbm', 'rb') as chunk_file:
    model_14 = chunk_file.read()

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part14.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_14 = b""
with open('my_regressor_part14.cbm', 'rb') as chunk_file:
    model_14 = chunk_file.read()
model_14 = io.BytesIO(model_14)
repo.create_file("my_regressor_part14.cbm", "commit message",model_14.getvalue())

################
model_15 = b""
with open('my_regressor_part15.cbm', 'rb') as chunk_file:
    model_15 = chunk_file.read()

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part15.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_15 = b""
with open('my_regressor_part15.cbm', 'rb') as chunk_file:
    model_15 = chunk_file.read()
model_15 = io.BytesIO(model_15)
repo.create_file("my_regressor_part15.cbm", "commit message",model_15.getvalue())

################
model_16 = b""
with open('my_regressor_part16.cbm', 'rb') as chunk_file:
    model_16 = chunk_file.read()

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part16.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_16 = b""
with open('my_regressor_part16.cbm', 'rb') as chunk_file:
    model_16 = chunk_file.read()
model_16 = io.BytesIO(model_16)
repo.create_file("my_regressor_part16.cbm", "commit message",model_16.getvalue())

################
model_17 = b""
with open('my_regressor_part17.cbm', 'rb') as chunk_file:
    model_17 = chunk_file.read()

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part17.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_17 = b""
with open('my_regressor_part17.cbm', 'rb') as chunk_file:
    model_17 = chunk_file.read()
model_17 = io.BytesIO(model_17)
repo.create_file("my_regressor_part17.cbm", "commit message",model_17.getvalue())

################
model_18 = b""
with open('my_regressor_part18.cbm', 'rb') as chunk_file:
    model_18 = chunk_file.read()

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part18.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_18 = b""
with open('my_regressor_part18.cbm', 'rb') as chunk_file:
    model_18 = chunk_file.read()
model_18 = io.BytesIO(model_18)
repo.create_file("my_regressor_part18.cbm", "commit message",model_18.getvalue())

################
model_19 = b""
with open('my_regressor_part2.cbm', 'rb') as chunk_file:
    model_19 = chunk_file.read()

repo = g.get_user().get_repo('Porsche')
all_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir" or file_content.type == "csv" :
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


git_prefix = 'Porsche/'
git_file = 'my_regressor_part19.cbm'
### THIS PART WORKS ###
contents = repo.get_contents(git_file)
repo.delete_file(contents.path, "deleting old file", contents.sha, branch="main")

model_19 = b""
with open('my_regressor_part19.cbm', 'rb') as chunk_file:
    model_19 = chunk_file.read()
model_19 = io.BytesIO(model_19)
repo.create_file("my_regressor_part19.cbm", "commit message",model_19.getvalue())

