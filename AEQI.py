"""
AEQI Computations
"""

"""
    TODO:faire continental
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unidecode

def add_continent(df):
    country_region = pd.read_csv('data/country_region.csv',index_col=0,sep=';')
    region_list=[]
    for uni in df.itertuples():
        ctr = uni.Index
        region_list.append(country_region.loc[ctr.upper(),'Region'])
    df.loc[:,"Continent"] = region_list
    return df

country_dict = { #This dict manages different names given by different countries
    'Hong Kong Sar': 'Hong Kong',
    'Hong Kong': 'Hong Kong',
    'China-Hong Kong': 'Hong Kong',
    'Iran, Islamic Republic Of': 'Iran',
    'Iran': 'Iran',
    'Brunei': 'Brunei',
    'Brunei Darussalam': 'Brunei',
    'China': 'China',
    'China (Mainland)': 'China',
    'China-Macau': 'Macau',
    'Macao': 'Macau',
    'Macau Sar': 'Macau',
    'China-Taiwan': 'Taiwan',
    'Taiwan': 'Taiwan',
    'Palestine': 'Palestine',
    'Palestinian Territory, Occupied': 'Palestine',
    'Russia': 'Russia',
    'Russian Federation': 'Russia'
}


for ctr_cont in ['country']:#,'Continent']:
    aeqi_dataframes = [] #This array will contain 3 dataframes, 1 for each ranking
    for rkg in ['THE','QS','ARWU']:
        rkg_db = pd.read_csv("data/"+rkg+"-data-aeqi-ready.csv")
        rkg_db.loc[:,"Pondered Rank"] = rkg_db.loc[:,"n_students"]*rkg_db.loc[:,"Rank"]
        rkg_db.loc[:,"Pondered Score"] = rkg_db.loc[:,"n_students"]*rkg_db.loc[:,"scores_overall_calc"]
        
        temp_aeqi_df = rkg_db[[ctr_cont,"Pondered Rank","Pondered Score","n_students"]].groupby(ctr_cont).sum()
        temp_aeqi_df.loc[:,rkg+" RAEQI"] = temp_aeqi_df.loc[:,"Pondered Rank"] / temp_aeqi_df.loc[:,"n_students"]
        temp_aeqi_df.loc[:,rkg+" AEQI"] = temp_aeqi_df.loc[:,"Pondered Score"] / temp_aeqi_df.loc[:,"n_students"]
        
        aeqi_dataframes += [temp_aeqi_df.loc[:,[rkg+" RAEQI",rkg+" AEQI"]]]
        
    aeqi = pd.concat(aeqi_dataframes,axis=1)
    
    if ctr_cont=='country':
        aeqi['Country'] = aeqi.index.map(country_dict)
        aeqi['Country'] = aeqi['Country'].fillna(aeqi.index.to_series())
        aeqi = aeqi.groupby('Country').first()
        
        aeqi = add_continent(aeqi)
        
        #We decided to only keep aeqi
        aeqi[["THE AEQI","QS AEQI","ARWU AEQI",'Continent']].to_excel('results/aeqi_'+ctr_cont.lower()+'.xlsx')
    else:
        aeqi[["THE AEQI","QS AEQI","ARWU AEQI"]].to_excel('results/aeqi_'+ctr_cont.lower()+'.xlsx')
#correlations
aeqi.loc[:,'THE AEQI rank'] = aeqi.loc[:,"THE AEQI"].rank(method='min',ascending=False)
aeqi.loc[:,'QS AEQI rank'] = aeqi.loc[:,"QS AEQI"].rank(method='min',ascending=False)
aeqi.loc[:,'ARWU AEQI rank'] = aeqi.loc[:,"ARWU AEQI"].rank(method='min',ascending=False)

aeqi.loc[:,'THE RAEQI rank'] = aeqi.loc[:,"THE RAEQI"].rank(method='min')

rank_corr_tab = aeqi[["THE AEQI","QS AEQI","ARWU AEQI"]].corr(method='spearman')

#################################################################################
#
#   Computing various regional AEQI
#
#################################################################################
#European union
non_UE = ['United Kingdom','Ukraine','Switzerland','Serbia','Russia','Norway','Northern Cyprus',
      'Montenegro','Iceland','Bosnia And Herzegovina','Belarus']
for rkg in ['THE','QS','ARWU']:
    rkg_db = pd.read_csv("data/"+rkg+"-data-aeqi-ready.csv")
    rkg_db.loc[:,"Pondered Score"] = rkg_db.loc[:,"n_students"]*rkg_db.loc[:,"scores_overall_calc"]
    
    
    rkg_db = rkg_db.loc[(rkg_db['Continent']=='EUROPE') & (~rkg_db['country'].isin(non_UE)),:]
    rkg_sum = rkg_db.sum()
    print('UE',' :: ',rkg,' :: ',rkg_sum['Pondered Score']/rkg_sum["n_students"])

print('')
#Belgium - FWB
VO = ['Ghent University','KU Leuven','University of Antwerp','Vrije Universiteit Brussel (VUB)','Vrije Universiteit Brussel','Hasselt University']
for rkg, uni_index in zip(['THE','QS','ARWU'],['institution_name','University','univNameEn']):
    rkg_db = pd.read_csv("data/"+rkg+"-data-aeqi-ready.csv")
    rkg_db.loc[:,"Pondered Score"] = rkg_db.loc[:,"n_students"]*rkg_db.loc[:,"scores_overall_calc"]
    
    rkg_db = rkg_db.loc[(rkg_db['country'].str.upper()=='BELGIUM') & (~rkg_db[uni_index].str.upper().isin([x.upper() for x in VO])),:]
    rkg_sum = rkg_db.sum()
    print('FWB',' :: ',rkg,' :: ',rkg_sum['Pondered Score']/rkg_sum["n_students"])
print('')
for rkg, uni_index in zip(['THE','QS','ARWU'],['institution_name','University','univNameEn']):
    rkg_db = pd.read_csv("data/"+rkg+"-data-aeqi-ready.csv")
    rkg_db.loc[:,"Pondered Score"] = rkg_db.loc[:,"n_students"]*rkg_db.loc[:,"scores_overall_calc"]
    
    rkg_db = rkg_db.loc[(rkg_db['country'].str.upper()=='BELGIUM') & (rkg_db[uni_index].str.upper().isin([x.upper() for x in VO])),:]
    rkg_sum2 = rkg_db.sum()
    print('VO',' :: ',rkg,' :: ',rkg_sum2['Pondered Score']/rkg_sum2["n_students"])


for rkg in ['THE','QS','ARWU']:
    rkg_db = pd.read_csv("data/"+rkg+"-data-aeqi-ready.csv")
    plt.plot(rkg_db["scores_overall_calc"][:1000].sort_values(ascending=False).to_numpy(),label=rkg)
plt.ylabel('Score') 
plt.legend()
plt.show()
