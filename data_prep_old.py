"""
Data cleaning and data formatting

@author: Sacha Dehoorne
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unidecode

#------------------------------------------------------------------------------
#   USEFUL FUNCTIONS
#------------------------------------------------------------------------------

"""
    col is a dataframe column containing HEI names 
    returns a normalized column, meaning we remove common words to make the databases 
        correspond even with disparities such as language or encoding
"""
def normalizeName(col):
    to_replace = ['of ','the ', 'de ', 'uni\S*','and ','& ','degli studi di ']
    norm = col.str.lower()
    norm = norm.str.split(' \(').str[0].str.split(' -').str[0]
    for rep in to_replace:
        norm = norm.str.replace(rep,' ', case=False, regex=True)
    norm = norm.str.strip().str.replace('\s{2,}',' ',regex=True)
    norm = norm.apply(unidecode.unidecode)
    return norm


#------------------------------------------------------------------------------
#    IMPORTING DATA
#------------------------------------------------------------------------------


the_all_years = pd.read_excel("data/the_bdd_2011-2023.xlsx")
arwu_all_years = pd.read_excel("data/ARWU-BDD-à partir de 2004.xlsx")
qs_all_years = pd.read_excel("data/qs_bdd-2012-2023.xlsx")


#QS Exclu : array containing names of universities appearing in QS but not in THE
with open("data/QS_exclu.txt",encoding='utf-8') as file:
    qs_exclu = [line.rstrip() for line in file]
    
#QS Exclu : array containing names of universities appearing in ARWU but neither in THE nor in QS
with open("data/ARWU_exclu.txt",encoding='utf-8') as file:
    arwu_exclu = [line.rstrip() for line in file]
    
#------------------------------------------------------------------------------
#    TRANSFORMING DATAFRAMES TO MAKE CORRESPONDANCE BETWEEN HEI LATER
#------------------------------------------------------------------------------

the_all_years.loc[:,"ReducedName"] = normalizeName(the_all_years.loc[:,'institution_name'])
the_all_years.index = the_all_years.loc[:,"institution_name"]

qs_all_years.loc[:,"ReducedName"] = normalizeName(qs_all_years.loc[:,'University'])

arwu_all_years.loc[:,"ReducedName"] = normalizeName(arwu_all_years.loc[:,"univNameEn"])

#only keep the year's we"ll work on
the = the_all_years.loc[the_all_years['year'] == 2022].copy()
qs = qs_all_years.loc[qs_all_years['Year'] == 2022].copy()
arwu = arwu_all_years.loc[arwu_all_years['year'] == 2021].copy()

#identify HEI that are in qs and arwu but not THE
arwu_qs_exclu = []
for uni in arwu.itertuples():
    uni_name = uni.ReducedName
    if len(the.loc[the["ReducedName"] == uni_name,:]) == 0:
        if len(qs.loc[qs["ReducedName"] == uni_name,:]) > 0:
            arwu_qs_exclu.append(uni.univNameEn)
            
#------------------------------------------------------------------------------
#    COMPUTING Rank and Score for THE Ranking
#------------------------------------------------------------------------------

the.loc[:,"the_wur_score"] = 0.3*the.loc[:,"citation_score"]+0.025*the.loc[:,"industry_score"]+0.075*the.loc[:,"international_outlook_score"] \
                        + 0.3*the.loc[:,"research_score"] + 0.3*the.loc[:,"teaching_score"]
the.loc[:,"THE Rank"] = the.loc[:,"the_wur_score"].rank(method='min',ascending=False)


#------------------------------------------------------------------------------
#    COMPUTING Rank and Score for THE Ranking
#------------------------------------------------------------------------------

all_rankings = the.loc[:,["ReducedName","country","THE Rank","stats_number_students"]].copy()
all_rankings.index = the.loc[:,"institution_name"].copy()
all_rankings = all_rankings.sort_values("THE Rank")

#------------------------------------------------------------------------------
#    Create a new dataframe with the respective ranks in each rankings. (only considering the n first HEI)
#------------------------------------------------------------------------------

n_first = 1000

for uni in qs.itertuples():
    uni_name = uni.ReducedName
    
    if uni.University not in qs_exclu:
        #For Uni in THE & QS
        if uni_name=='istanbul': #special case
            all_rankings.loc['Istanbul University',"QS Rank"] = uni.Rank
        else: 
            #if there is only 1 uni corresponding
            if len(all_rankings.loc[all_rankings['ReducedName'] == uni_name].to_numpy()) == 1:
                all_rankings.loc[all_rankings['ReducedName'] == uni_name,"QS Rank"] = uni.Rank
        
            else:
                #for multiple uni corresponding the name we differentiate them using the country
                all_rankings.loc[(all_rankings['country']==uni.Location.title()) & 
                         (all_rankings['ReducedName'] == uni_name),'QS Rank'] = uni.Rank

    else:
        #for uni in QS but not THE
        try:
            new_uni = {'QS Rank':uni.Rank, 'ReducedName':uni_name, 'country':uni.Location.title()}
            all_rankings = all_rankings.append(pd.DataFrame([new_uni],index=[uni.University],columns=all_rankings.columns))
        except:
            print('probleme')                

for uni in arwu.itertuples():
    uni_name = uni.ReducedName
    if uni.univNameEn in arwu_exclu and uni_name not in arwu_qs_exclu:
        #Uni only in ARWU
        new_uni = {'ARWU Rank':uni.Rank+1, 'ReducedName':uni_name, 'country':uni.region.title()}
        all_rankings = all_rankings.append(pd.DataFrame([new_uni],index=[uni.univNameEn],columns=all_rankings.columns))
    else: #in THE
        all_rankings.loc[all_rankings['ReducedName'] == uni_name,"ARWU Rank"] = uni.Rank+1
            
#make rank >1000 = na as a way to not count them
for ind in ["THE Rank","QS Rank","ARWU Rank"]:
    all_rankings.loc[all_rankings[ind] > 1000, [ind]] = np.nan
    
#delete irrelevant rows
all_rankings = all_rankings[(all_rankings["THE Rank"].notna()) | (all_rankings["QS Rank"].notna()) |
                            (all_rankings["ARWU Rank"].notna())]
#all_rankings.index = all_rankings.loc[:,"institution_name"]
all_rankings = all_rankings.drop("ReducedName",axis=1)

#------------------------------------------------------------------------------
#   Add column for continent of HEI
#------------------------------------------------------------------------------

country_region = pd.read_csv('data/country_region.csv',index_col=0,sep=';')
region_list=[]
for uni in all_rankings.itertuples():
    #uniform country:
    ctr = uni.country
    for i,j in zip(['Brunei Darussalam','China (Mainland)','Hong Kong Sar'],['Brunei','China','Hong Kong']):
        if uni.country == i:
            all_rankings.loc[uni.Index,'country'] = j
            ctr = j
    region_list.append(country_region.loc[ctr.upper(),'Region'])
all_rankings.loc[:,"Region"] = region_list

all_rankings.to_csv('allrankings.csv')










