import plotly.express as px
import plotly.io as io
io.renderers.default='browser'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
plt=reload(plt)
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import geopandas as gpd
import pycountry
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

def format_wb(col):
    return pd.to_numeric(col.str.split('(').str[0],errors='coerce')

aeqi = pd.read_excel("results/aeqi_country.xlsx",index_col=0)
aeqi_continent = pd.read_excel("results/aeqi_continent.xlsx",index_col=0)

###############################################################################
#
#   Some plots
#
###############################################################################

for rkg in ['THE','QS','ARWU']:
    plt.plot(aeqi[rkg+' AEQI'].sort_values(ascending=False).to_numpy(),label=rkg)
plt.ylabel('Score') 
plt.legend()
plt.show()

###############################################################################
#
#   Importing regressors
#
###############################################################################
educ = pd.read_excel('data/world-data/Educ_data.xlsx',index_col=0).transpose().apply(format_wb)
resea = pd.read_excel('data/world-data/Research_data.xlsx',index_col=0).apply(format_wb)

gini = pd.read_excel("data/world-data/gini.xlsx", index_col=0)
pop = pd.read_excel("data/world-data/population.xls", index_col=0)

educ['PISA mean'] = educ.iloc[:,1]+educ.iloc[:,2]+educ.iloc[:,3]
educ = educ.drop(educ.columns[1:4], axis=1)

aeqi_indic = aeqi.copy()
ind_l = np.append(np.append(educ.columns.values,['Gini','LPop']),resea.columns.values)

for c in aeqi_indic.itertuples():
    ctr =c.Index
    try:
        for col_names in educ.columns.values:
            aeqi_indic.loc[c.Index,col_names] = educ.loc[ctr,col_names]
        for col_names in resea.columns.values:
            aeqi_indic.loc[c.Index,col_names] = resea.loc[ctr,col_names]
        aeqi_indic.loc[c.Index,'Gini'] = gini.loc[ctr,'LAST']
        #aeqi_indic.loc[c.Index,'Pop'] = pop.loc[ctr,'2022']
        aeqi_indic.loc[c.Index,'LPop'] = np.log(pop.loc[ctr,'2022'])
       
    except: #TAIWAN and Palesine
        aeqi_indic.loc[c.Index,col_names] = np.nan
        #print(ctr)
        
      
###############################################################################
#
#   Multiregression - OLS
#
###############################################################################

plot_detailled = False
i=-1
for rkg in ['THE','QS','ARWU']:
    i+=1
    aeqi_indic = aeqi_indic[aeqi_indic[rkg+' AEQI'].notna()]
    
    Y=aeqi_indic[rkg+' AEQI']
    X=aeqi_indic[ind_l]#.iloc[:,[0]]
    
    X.iloc[:,12] =X.iloc[:,12] / np.exp(X.iloc[:,10])
    X.iloc[:,11] =X.iloc[:,11] * X.iloc[:,16] /100
    X.iloc[:,3] = X.iloc[:,3] * X.iloc[:,16] /100
    
    col_aliases = ['Graduation rate','Enroll. rate','Funding per student','Edu Exp. (per capita)',
                   'HCI','Private','Inbound','Outbound','PISA mean','Gini','LPop','R&D Exp (per capita)','Patents per capita','# R&D Researchers',
                   'Sci articles','GDP (per capita, ppp)','GDP (per capita)']
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=col_aliases, index=X.index)
    corr_x = np.corrcoef(X_imputed.T)
    #X_imputed = X_imputed.iloc[:,[2,5,6,7,11,12,15]] #2,5,6,7,11,12,15
    X_imputed = X_imputed.iloc[:,[2,10,11]]
    
        
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_imputed.columns
      
     #calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X_imputed.values, i)
                              for i in range(len(X_imputed.columns))]
      
    print(vif_data)
    
    X_imputed = sm.add_constant(X_imputed)

    model = sm.OLS(Y, X_imputed)
    res = model.fit()
    #print(rkg ,' : ')
    print(res.summary())
    
    
    if plot_detailled:
        x=X_imputed['Patents per capita']
        y=Y
        plt.scatter(x,y,s=2)
        #plt.xlim([0,0.00025])
        m, b = np.polyfit(x,y, 1)
        x_plt = np.linspace(x.min(),x.max(),1000)
        plt.plot(x_plt, m*x_plt+b,c='orange',linestyle='dashed')
        plt.ticklabel_format(axis='x', style='sci',scilimits=(-3,3))
        plt.ylabel('AEQI (THE)')
        plt.xlabel('Patents per capita')
        plt.show()
        #p-value =
        
        x=np.log(X_imputed['Patents per capita'])
        y=Y
        plt.scatter(x,y,s=2)
        #plt.xlim([0,0.00025])
        m, b = np.polyfit(x,y, 1)
        x_plt = np.linspace(x.min(),x.max(),1000)
        plt.plot(x_plt, m*x_plt+b,c='orange',linestyle='dashed')
        plt.ticklabel_format(axis='x', style='sci',scilimits=(-3,3))
        plt.ylabel('AEQI (THE)')
        plt.xlabel('Patents per capita')
        plt.show()
        #p-value =
        
        
        x=X_imputed['Private']
        y=Y
        plt.scatter(x,y,s=2)
        #plt.xlim([0,0.00025])
        m, b = np.polyfit(x,y, 1)
        x_plt = np.linspace(x.min(),x.max(),1000)
        plt.plot(x_plt, m*x_plt+b,c='orange',linestyle='dashed')
        plt.ylabel('AEQI (THE)')
        plt.xlabel('Private')
        plt.show()
        
        x=X_imputed['Funding per student']
        y=Y
        plt.scatter(x,y,s=2)
        #plt.xlim([0,0.00025])
        m, b = np.polyfit(x,y, 1)
        x_plt = np.linspace(x.min(),x.max(),1000)
        plt.plot(x_plt, m*x_plt+b,c='orange',linestyle='dashed')
        plt.ylabel('AEQI (THE)')
        plt.xlabel('Funding per student')
        plt.show()
        
        x=X_imputed['R&D Exp (per capita)']
        y=Y
        plt.scatter(x,y,s=2)
        #plt.xlim([0,0.00025])
        m, b = np.polyfit(x,y, 1)
        x_plt = np.linspace(x.min(),x.max(),1000)
        plt.plot(x_plt, m*x_plt+b,c='orange',linestyle='dashed')
        plt.ylabel('AEQI (THE)')
        plt.xlabel('R&D Exp (per capita)')
        plt.show()
    
        x=X_imputed['Inbound']
        y=Y
        plt.scatter(x,y,s=2)
        #plt.xlim([0,0.00025])
        m, b = np.polyfit(x,y, 1)
        x_plt = np.linspace(x.min(),x.max(),1000)
        plt.plot(x_plt, m*x_plt+b,c='orange',linestyle='dashed')
        plt.ylabel('AEQI (THE)')
        plt.xlabel('R&D Exp (per capita)')
        plt.show()
    
        print(X_imputed['Inbound'].describe())
        print(X_imputed['Outbound'].describe())
        print(X_imputed['GDP (per capita, ppp)'].describe())

###############################################################################
#
#   Map
#
###############################################################################
"""
for rkg in ['THE','QS','ARWU']:
    aeqi[rkg+" AEQI Rank"] = aeqi[rkg+" AEQI"].rank(method='min',ascending=False)
    aeqi_continent[rkg+" AEQI Rank"] = aeqi_continent[rkg+" AEQI"].rank(method='min',ascending=False)
"""

aeqi_tern = aeqi.copy().dropna()
def mean_norm(df_input):
    return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

def minmax_norm(df_input):
    return (df_input - df_input.min()) / ( df_input.max() - df_input.min())

def quantile_norm(df_input):
    sorted_df = pd.DataFrame(np.sort(df_input.values,axis=0), index=df_input.index, columns=df_input.columns)
    mean_df = sorted_df.mean(axis=1)
    mean_df.index = np.arange(1, len(mean_df) + 1)
    quantile_df =df_input.rank(method="min").stack().astype(int).map(mean_df).unstack()
    return(quantile_df)

aeqi_tern[["THE AEQI","QS AEQI","ARWU AEQI"]] = minmax_norm(aeqi_tern[["THE AEQI","QS AEQI","ARWU AEQI"]])

fig = px.scatter_ternary(aeqi_tern.reset_index(), 
                         a="THE AEQI", b="QS AEQI", c="ARWU AEQI",color='Continent',hover_name='Country')
fig.write_html("ternary_ctr.html")
#fig.show()

#fig = px.scatter_ternary(aeqi_continent.reset_index(), 
#                         a="THE AEQI Rank", b="QS AEQI Rank", c="ARWU AEQI Rank",color='Continent',hover_name='Continent')
##ig.write_html("ternary_continent.html")
#fig.show()




aeqi_map =  aeqi.copy().dropna()
aeqi_map[["THE AEQI","QS AEQI","ARWU AEQI"]] = mean_norm(aeqi_map[["THE AEQI","QS AEQI","ARWU AEQI"]])

for rkg in ['THE','QS','ARWU']:
    plt.plot(aeqi_map[rkg+' AEQI'].sort_values(ascending=False).to_numpy(),label=rkg)
plt.ylabel('Score') 
plt.legend()
plt.show()


aeqi_map['AEQI Mean'] = aeqi_map.mean(axis=1)
for rkg in ['THE','QS','ARWU']:
    aeqi_map[rkg+ ' AEQI diff'] = aeqi_map[rkg+' AEQI'] - aeqi_map['AEQI Mean']



def standardize_country_name(name):
    if name == 'Russia':
        return pycountry.countries.lookup('RUS').name
    if name == 'Iran':
        return pycountry.countries.lookup('IRN').name
    try:
        # Try to get the official country name
        return pycountry.countries.lookup(name).name
    except LookupError:
        # If the name is not found, return the original name
        print(name)
        return name


# add a column for the highest value column
aeqi_map['highest_col'] = aeqi_map[["THE AEQI diff","QS AEQI diff","ARWU AEQI diff"]].idxmax(axis=1)

# add a column for the highest value and standardised
aeqi_map[["THE AEQI diff","QS AEQI diff","ARWU AEQI diff"]] = aeqi_map[["THE AEQI diff","QS AEQI diff","ARWU AEQI diff"]]#.apply(lambda x: (x-x.min()) / (x.max()-x.min()))
for i, row in aeqi_map.iterrows():
    aeqi_map.loc[i,'highest_val'] = aeqi_map.loc[i,aeqi_map.loc[i,'highest_col']]

aeqi_map['country'] = aeqi_map.index
aeqi_map['country'] = aeqi_map['country'].apply(standardize_country_name)

# load GeoDataFrame with country geometries
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world['name'] = world['iso_a3'].apply(standardize_country_name)

# merge data with world GeoDataFrame
world = world.merge(aeqi_map, how='left', left_on='name', right_on='country')

colormaps = ['Blues', 'Reds', 'Greens']

# Function to convert column name and value to a color
def get_color(row):
    if pd.isna(row['highest_col']):
        return 'darkgrey'
    else:
        col_index = ["THE AEQI diff","QS AEQI diff","ARWU AEQI diff"].index(row['highest_col'])
        colormap = colormaps[col_index]
        return plt.get_cmap(colormap)(row['highest_val'])
    
# Apply function to dataframe
world['color'] = world.apply(get_color, axis=1)

# plot the world dataframe
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
world.boundary.plot(ax=ax, linewidth=1)
world.plot(color=world['color'], linewidth=0.8, ax=ax, edgecolor='0.8')

# Create legend
red_patch = mpatches.Patch(color='red', label='QS')
blue_patch = mpatches.Patch(color='blue', label='THE')
green_patch = mpatches.Patch(color='green', label='ARWU')
plt.legend(handles=[red_patch, blue_patch, green_patch], loc='lower left', fontsize = 20)

plt.show()

###############################################################################
#
#   Distribution of AEQI
#
###############################################################################
 

aeqi_map2 =  aeqi.copy().drop('ARWU AEQI',axis=1)#.dropna()
aeqi_map2[["THE AEQI","QS AEQI"]] = mean_norm(aeqi_map2[["THE AEQI","QS AEQI"]])

for rkg in ['THE','QS']:
    plt.plot(aeqi_map2[rkg+' AEQI'].sort_values(ascending=False).to_numpy(),label=rkg)
plt.ylabel('Score') 
plt.legend()
plt.show()


aeqi_map2['AEQI Mean'] = aeqi_map2.mean(axis=1)
for rkg in ['THE','QS']:
    aeqi_map2[rkg+ ' AEQI diff'] = aeqi_map2[rkg+' AEQI'] - aeqi_map2['AEQI Mean']



def standardize_country_name(name):
    if name == 'Russia':
        return pycountry.countries.lookup('RUS').name
    if name == 'Iran':
        return pycountry.countries.lookup('IRN').name
    try:
        # Try to get the official country name
        return pycountry.countries.lookup(name).name
    except LookupError:
        # If the name is not found, return the original name
        print(name)
        return name


# add a column for the highest value column
aeqi_map2['highest_col'] = aeqi_map2[["THE AEQI diff","QS AEQI diff"]].idxmax(axis=1)

# add a column for the highest value and standardised
aeqi_map2[["THE AEQI diff","QS AEQI diff"]] = aeqi_map2[["THE AEQI diff","QS AEQI diff"]]#.apply(lambda x: (x-x.min()) / (x.max()-x.min()))
for i, row in aeqi_map2.iterrows():
    aeqi_map2.loc[i,'highest_val'] = aeqi_map2.loc[i,aeqi_map2.loc[i,'highest_col']]

aeqi_map2['country'] = aeqi_map2.index
aeqi_map2['country'] = aeqi_map2['country'].apply(standardize_country_name)

# load GeoDataFrame with country geometries
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world['name'] = world['iso_a3'].apply(standardize_country_name)

# merge data with world GeoDataFrame
world = world.merge(aeqi_map2, how='left', left_on='name', right_on='country')

colormaps = ['Blues', 'Reds', 'Greens']

# Function to convert column name and value to a color
def get_color(row):
    if pd.isna(row['highest_col']):
        return 'darkgrey'
    else:
        col_index = ["THE AEQI diff","QS AEQI diff"].index(row['highest_col'])
        colormap = colormaps[col_index]
        return plt.get_cmap(colormap)(row['highest_val'])
    
# Apply function to dataframe
world['color'] = world.apply(get_color, axis=1)

# plot the world dataframe
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
world.boundary.plot(ax=ax, linewidth=1)
world.plot(color=world['color'], linewidth=0.8, ax=ax, edgecolor='0.8')

# Create legend
red_patch = mpatches.Patch(color='red', label='QS')
blue_patch = mpatches.Patch(color='blue', label='THE')
plt.legend(handles=[red_patch, blue_patch], loc='lower left', fontsize = 20)

plt.show()
    
    
    
    
    

