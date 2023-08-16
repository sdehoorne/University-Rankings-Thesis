"""
Data cleaning and data formatting

@author: Sacha Dehoorne
"""
import pandas as pd
import numpy as np
import unidecode
import re
from bs4 import BeautifulSoup
import requests

#normalize university names for qs text file containing stu count
def normalize_string_qs(str):
    
    str = str.replace('Å ','S')
    str=str.lower()
    str = str.replace('soochow university (taiwan)','soochow(taiwan)')
    str = str.replace('palermo (up)','palermo(up)')
    str = str.replace('university–camden','ucamden')
    
    
    str = str.split(' -')[0].split(' (')[0]#.split(', ')[0]
    
    to_replace = ['of ','the ', 'de ',' and ','& ','degli studi di ',',','at ',' in ','s.d. ']
    for rep in to_replace:
        str = str.replace(rep,' ')
    str = re.sub('(?i)uni\S*',' u ',str)
    str = re.sub('(?i)\s{2,}',' ',str).strip()
    
    str = unidecode.unidecode(str).strip()
    
    encoding_errors=[['aoe','u'],['a"','o'],['a%0','e'],['af','a'],['a-','o'],["a'",'o'],['aEUR','a'],['a(tm)',"'"],['a*','*'],
                     ['a^','e'],['e>>',"`"],['a,,','a'],['a~','o'],['a++','c'],['a 1/2 ','z']]
    for rep in encoding_errors:
        str = str.replace(rep[0],rep[1]).strip()
    
    diff_names = [['national u ireland galway','u galway'],['u paris cite','u paris'],['u ulm','ulm u'],['westfalische wilhelms-',''],
                  ['u des saarlandes','saarland u'], ['bogor agricultural u','ipb u'],['paris-pantheon-assas','paris 2 pantheon-assas'],
                  ['comillas pontifical u','u pontificia comillas'],['pontificia u catolica do rio janeiro','pontifacia u catolica do rio janeiro'],
                  ['ibaoez','ibanez'],['repasblica','republica'],['valparaaso','valparaiso'],['brasalia','brasilia'],['gdaask','gdansk'],
                  ['poznaa','poznan'],['bolavar','bolivar'],['nantes u','u nantes'],
                  ['academician y.a. buketov karaganda u','karaganda buketov u'],['a" sggw',''],['al quds u arab u jerusalem','al-quds u'],
                  ['s. toraighyrov pavlodar state u','toraighyrov u'],['dokuz eylul universitesi','dokuz eylul u']]
    for rep in diff_names:
        str = str.replace(rep[0],rep[1])
    
    return str.strip()

#normalize university names for ARWU text file containing stu count
def normalize_string_arwu(str):
    str = str.lower()
    str = unidecode.unidecode(str).strip()
    encoding_errors=[['-',' - '],['a(c)','e'],["a'",'o'],['a€"','-']]
    
    to_replace = ['of ','the ', 'de ',' and ','& ','degli studi di ',',','at ',' in ','s.d. ']
    for rep in to_replace:
        str = str.replace(rep,' ')
        
    for rep in encoding_errors:
        str = str.replace(rep[0],rep[1]).strip()
    
    str = re.sub('(?i)\s{2,}',' ',str).strip()
    return str.strip()



def find_on_web(query):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"

    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, 'html.parser')

    # Trying to find answer in different formats and locations
    selectors = ['div.Z0LcW', 'div.kpd-ans', 'div.zCubwf', 'div.vk_c', 'div.g']

    for selector in selectors:
        result = soup.select_one(selector)
        if result:
            return ' '.join(result.stripped_strings)

    return 'No result found'

def add_continent(df):
    country_region = pd.read_csv('data/country_region.csv',index_col=0,sep=';')
    region_list=[]
    for uni in df.itertuples():
        ctr = uni.country
        for i,j in zip(['Brunei Darussalam','China (Mainland)','Hong Kong Sar'],['Brunei','China','Hong Kong']):
            if uni.country == i:
                df.loc[uni.Index,'country'] = j
                ctr = j
        region_list.append(country_region.loc[ctr.upper(),'Region'])
    df.loc[:,"Continent"] = region_list
    return df

#------------------------------------------------------------------------------
#    IMPORTING QS DATA
#------------------------------------------------------------------------------


qs_stu = []
with open("data/QS-students.txt",encoding='utf-8') as file:
    for line in file:
        line_txt = line.rstrip().split('::')
        qs_stu += [[normalize_string_qs(line_txt[0].upper()),int(line_txt[1].replace(",", ""))]]
        
qs_stu = np.array(qs_stu)        
qs_stu = pd.DataFrame(qs_stu[:,1], index = qs_stu[:,0])

qs_all_years = pd.read_excel("data/QS-data.xlsx")
qs = qs_all_years.loc[qs_all_years['Year'] == 2022].copy()

i=0
for uni in qs.itertuples():
    if normalize_string_qs(uni.University) != 'u los andes':
            qs.loc[uni.Index,'n_students'] = qs_stu.loc[normalize_string_qs(uni.University),0]
    else:
        if  uni.University == 'UNIVERSIDAD DE LOS ANDES':
            qs.loc[uni.Index,'n_students'] = 15312
        elif uni.University.split(' - ')[1] == 'CHILE':
            qs.loc[uni.Index,'n_students'] = 9946
        else:
            qs.loc[uni.Index,'n_students'] = 24372

#qs['Continent'] = qs['Region'].replace('NORTH AMERICA','AMERICA').replace('LATIN AMERICA','AMERICA').str.title()
qs['country']=qs['Location'].str.title()

qs = add_continent(qs)

qs[["country",'Continent','Rank','University','scores_overall_calc','n_students']].to_csv("data/QS-data-aeqi-ready.csv")

#------------------------------------------------------------------------------
#    IMPORTING ARWU DATA
#------------------------------------------------------------------------------

#the_all_years = pd.read_excel("data/the_bdd_2011-2023.xlsx")
#arwu_all_years = pd.read_excel("data/ARWU-BDD-à partir de 2004.xlsx")
i=0
j=0
arwu_stu = []
with open("data/ARWU-students.txt",encoding='utf-8') as file:
    for line in file:
        line_txt = line.rstrip().split('::')
        if line_txt[1] == '/':
            try:
                stu_count = qs_stu.loc[normalize_string_qs(line_txt[0].upper()),0]
                
            except:
                stu_count = find_on_web(line_txt[0]+' number of students')
                #print(line_txt[0],' : ',stu_count[0])
                if stu_count[0] in ['1','2','3','4','5','6','7','8','9']:
                    stu_count = stu_count.split(' ')[0]
                    stu_count = stu_count.replace(',', '').replace('.', '')
                    stu_count = int(stu_count)
                    j+=1
                else: 
                    stu_count = -1
                    i+=1
                    print(line_txt[0])
        else:
            stu_count = int(line_txt[1])
            
        arwu_stu += [[normalize_string_arwu(line_txt[0].upper()),stu_count]]
        
arwu_stu = np.array(arwu_stu)        
arwu_stu = pd.DataFrame(arwu_stu[:,1], index = arwu_stu[:,0])

arwu_all_years = pd.read_excel("data/ARWU-data.xlsx")
arwu = arwu_all_years.loc[arwu_all_years['year'] == 2021].copy()


for uni in arwu.itertuples():
    try:
        arwu.loc[uni.Index,'n_students'] = arwu_stu.loc[normalize_string_arwu(uni.univNameEn),0]
    except:
        print('ERROR')
print(i,j)

arwu.loc[:,'country'] = arwu.loc[:,'region']
arwu.loc[:,'Rank']=arwu.loc[:,'Rank']+1
arwu = add_continent(arwu)
arwu[["country",'Continent','univNameEn','Rank','scores_overall_calc','n_students']].to_csv("data/ARWU-data-aeqi-ready.csv")

#------------------------------------------------------------------------------
#    IMPORTING THE DATA
#------------------------------------------------------------------------------
the_all_years = pd.read_excel("data/THE-data.xlsx")
the = the_all_years.loc[the_all_years['year'] == 2022].copy()
the.loc[:,"scores_overall_calc"] = 0.3*the.loc[:,"citation_score"]+0.025*the.loc[:,"industry_score"]+0.075*the.loc[:,"international_outlook_score"] \
                        + 0.3*the.loc[:,"research_score"] + 0.3*the.loc[:,"teaching_score"]
                        
the.loc[:,"Rank"] = the.loc[:,"scores_overall_calc"].rank(method='min',ascending=False)
the.loc[:,'n_students'] = the.loc[:,'stats_number_students']

the = add_continent(the)

the[["country",'Continent','Rank','institution_name','scores_overall_calc','n_students']].to_csv("data/THE-data-aeqi-ready.csv")










