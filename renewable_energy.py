from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd 
import statsmodels.formula.api as sm
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import adfuller
from linearmodels import PanelOLS
from linearmodels import RandomEffects
from linearmodels.panel import FirstDifferenceOLS
from statsmodels.tsa.vector_ar.vecm import coint_johansen




plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})

df = pd.read_csv('/Users/soheilsaadatfar/Desktop/renewal_energy/renewable_energy_final.csv')


###################################################################################
#############################   Data  Preparation   ###############################
###################################################################################



#rename columns (variables)
df_new= df.rename(columns={'GDP, PPP (current international $) [NY.GDP.MKTP.PP.CD]': 'GDP_curr_ppp',
'Unemployment, total (% of total labor force) (national estimate) [SL.UEM.TOTL.NE.ZS]': 'Unemployment_total_percent',
'CO2 emissions (kt) [EN.ATM.CO2E.KT]': 'CO2_emission',
'GDP per unit of energy use (PPP $ per kg of oil equivalent) [EG.GDP.PUSE.KO.PP]': 'GDP_per_unit_energy_$',
'Energy use (kg of oil equivalent) per $1,000 GDP (constant 2017 PPP) [EG.USE.COMM.GD.PP.KD]': 'energy_use_per_1000GDP',
'Renewable energy consumption (% of total final energy consumption) [EG.FEC.RNEW.ZS]': 'REC_percent',
'Energy intensity level of primary energy (MJ/$2011 PPP GDP) [EG.EGY.PRIM.PP.KD]': 'EG_intensity_ppp_GDP',
'Gross fixed capital formation (current US$) [NE.GDI.FTOT.CD]': 'GFCF_current',
'Country Name': 'Country_Name', 'Country Code': 'Country_code',
'Time Code': 'Time_code'
})



####################### Attribute conversions (RECODING)  #######################

#convert date data type from float to int
#step 1: clean end of data footnote & empty space (and rename to df)
df= df_new.iloc[:-5]


#step 2: convert time to int
df.dtypes
df['Time'] = df['Time'].astype(int)


#convert country code to string
df['Country_code'] = df['Country_code'].astype(str)


#convert unemployment_total_percent object to float
df1 = df.copy()
df1['Unemployment_total_percent'] = pd.to_numeric(df1['Unemployment_total_percent'], errors='coerce')

#convert CO2_emission objeect to float
df1['CO2_emission'] = pd.to_numeric(df1['CO2_emission'], errors='coerce')

#convert GDP_per_unit_energy_$, object to float
df1['GDP_per_unit_energy_$'] = pd.to_numeric(df1['GDP_per_unit_energy_$'], errors='coerce')

#convert energy_use_per_1000GDP, object to float
df1['energy_use_per_1000GDP'] = pd.to_numeric(df1['energy_use_per_1000GDP'], errors='coerce')

#convert GFCF_current, object to float
df1['GFCF_current'] = pd.to_numeric(df1['GFCF_current'], errors='coerce')




######check for Nan (null)###

df1['GDP_curr_ppp'].isnull().values.any()
df1['GFCF_current'].isnull().values.any()
df1['REC_percent'].isnull().values.any()
df1['Unemployment_total_percent'].isnull().values.any()
df1['EG_intensity_ppp_GDP'].isnull().values.any()
df1['CO2_emission'].isnull().values.any()
df1['GDP_per_unit_energy_$'].isnull().values.any()
df1['energy_use_per_1000GDP'].isnull().values.any()




##### Clean up and even lenght of panel data ######
df1= df1[~df1['Country_Name'].isin(['Ukraine'])]
df1= df1[~df1['Time'].isin(['1990'])]
df1= df1[~df1['Time'].isin(['1991'])]
df1= df1[~df1['Time'].isin(['2015'])]
df1.head(10)



### replace 2 nan with mean #####
df1['Unemployment_total_percent'].fillna((df1['Unemployment_total_percent'].mean()), inplace=True)

######## remove Nan rows ###########

df1.dropna(axis=0,how='any',inplace=True)



#set index for dataframe (panel data)
df_indx = df1.set_index(['Country_Name','Time'])





df_bc= df1.pivot(index='Time', columns='Country_Name')





###############################################################################
#############################   Visualization    ##############################
###############################################################################


#GDP_curr_ppp Graph
gdpc = df1.pivot(index='Time', columns='Country_Name', values='GDP_curr_ppp')
gdpc.plot(title='GDP Current (PPP) by County')



#REC (renewable energy percentage)
REC_p= df.pivot(index='Time', columns='Country_Name', values='REC_percent')
REC_p.plot(title='Percent of Renewable Energy Used (By Countries)')





#Draw Plot (all attribute correlation)

ins_plot= sns.pairplot(df1)





####################################################################################
#############################   correlation matrix    ##############################
####################################################################################




corr = df_indx.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(
    corr,          
    mask=mask,    
    cmap=cmap,     
    annot=True,    
    vmax=.3,      
    vmin=-.3,    
    center=0,     
    square=True,  
    linewidths=.5, 
    cbar_kws={"shrink": .5}  
)









###################################################################################
#############################   Stationarity test   ###############################
###################################################################################


###GDP_curr_ppp####
x_df_gdp = df1['GDP_curr_ppp'].values
result= adfuller(x_df_gdp, autolag="AIC")
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')    

#Null Ho: time series is non-stationary
#Results:
#ADF Statistic: -4.859996001235914
#n_lags: 4.169731394548552e-05
#p-value: 4.169731394548552e-05
#Critial Values:
 #  1%, -3.4407890899441127
#Critial Values:
#   5%, -2.8661459340578905
#Critial Values:
#  10%, -2.5692228868342766

#p < alpha: reject Ho
#so there is evidence of stationary data

df_cad= df_indx.loc['Canada']
x_df_gdp_cad = df_cad['GDP_curr_ppp'].values
result= adfuller(x_df_gdp_cad, autolag="AIC")
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')    

#But here clearly if we analys the model by each country
#the results will be non stationary 
#P > alpha at .01, .05... 





#######Unemployment_total_percent stationarity#####
x_df_unemp = df1['Unemployment_total_percent'].values
result= adfuller(x_df_unemp, autolag="AIC")
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')   



#ADF Statistic: -5.620820508304043
#n_lags: 1.1462241891928024e-06
#p-value: 1.1462241891928024e-06
#Critial Values:
#   1%, -3.440873084361836
#Critial Values:
#   5%, -2.866182929627575
#Critial Values:
#   10%, -2.5692425987220577
#P < alpha so there is evidence of stationary data


#### CO2_emission stationarity ####
x_df_CO2 = df1['CO2_emission'].values
result= adfuller(x_df_CO2, autolag="AIC")
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')  

### Results: CO2
#ADF Statistic: -4.346871323717129
#n_lags: 0.0003681312971578547
#p-value: 0.0003681312971578547
#Critial Values:
#   1%, -3.4407890899441127
#Critial Values:
#   5%, -2.8661459340578905
#Critial Values:
#   10%, -2.5692228868342766
#P < alpha so there is evidence of stationary data



#### GDP_per_unit_energy_$  stationarity ####
x_df_GDP_eng = df1['GDP_per_unit_energy_$'].values
result= adfuller(x_df_GDP_eng, autolag="AIC")
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')  


##Results:
#ADF Statistic: -5.749527275581214
#n_lags: 6.011507216937968e-07
#p-value: 6.011507216937968e-07
#Critial Values:
#   1%, -3.4407890899441127
#Critial Values:
#   5%, -2.8661459340578905
#Critial Values:
#   10%, -2.5692228868342766
#P < alpha so there is evidence of stationary data



#### energy_use_per_1000GDP: stationarity #####
x_df_eng_per1000gdp = df1['energy_use_per_1000GDP'].values
result= adfuller(x_df_eng_per1000gdp, autolag="AIC")
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}') 

#results:
#ADF Statistic: -3.2487433520355062
#n_lags: 0.01733100002484599
#p-value: 0.01733100002484599
#Critial Values:
#   1%, -3.4408057813848147
#Critial Values:
#   5%, -2.866153285929353
#Critial Values:
#   10%, -2.569226804028561

# at critical value (alpha) -3.4409 variable is not staionary
# at critical value (alpha) -2.87   variable is staionary


### EG_intensity_ppp_GDP: stationarity ###
x_df_eng_intst = df1['EG_intensity_ppp_GDP'].values
result= adfuller(x_df_eng_intst, autolag="AIC")
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}') 


#Results:
#ADF Statistic: -3.3339454378499376
#n_lags: 0.013427641576433518
#p-value: 0.013427641576433518
#Critial Values:
#   1%, -3.4408057813848147
#Critial Values:
#   5%, -2.866153285929353
#Critial Values:
#   10%, -2.569226804028561

# at critical value (alpha) -3.4409 variable is not staionary
# at critical value (alpha) -2.87   variable is staionary



df1.head(5)

###GFCF_current: staionarity ###
x_df_GFCF = df1['GFCF_current'].values
result= adfuller(x_df_GFCF, autolag="AIC")
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}') 
#Results:
#ADF Statistic: -5.326919251538288
#n_lags: 4.814697994665934e-06
#p-value: 4.814697994665934e-06
#Critial Values:
#   1%, -3.4407890899441127
#Critial Values:
#   5%, -2.8661459340578905
#Critial Values:
#   10%, -2.5692228868342766




##################################################################################
############################   Cointegration test   ##############################
##################################################################################


ts.coint(df_indx.GDP_curr_ppp, df_indx.Unemployment_total_percent)

#coin_result: Ho: No cointegration exist
# (-4.752825473479189,
# 0.00045661863555135606,
# array([-3.91393635, -3.34586124, -3.05119964]))
#P-value < alpha 0.05 or 0.01: we reject H0:
#so there is evidence of cointegration

#p < alpha: reject Ho: No cointegration
#so there is evidence of cointegration



ts.coint(df_indx.GDP_curr_ppp, df_indx.GFCF_current)
ts.coint(df_indx.GDP_curr_ppp, df_indx.REC_percent)
ts.coint(df_indx.GDP_curr_ppp, df_indx.EG_intensity_ppp_GDP)
ts.coint(df_indx.GDP_curr_ppp, df_indx.CO2_emission)
ts.coint(df_indx.GDP_curr_ppp, df_indx['GDP_per_unit_energy_$'])
ts.coint(df_indx.GDP_curr_ppp, df_indx.energy_use_per_1000GDP)
ts.coint(df_indx.GFCF_current, df_indx.REC_percent)
ts.coint(df_indx.GFCF_current, df_indx.Unemployment_total_percent)
ts.coint(df_indx.GFCF_current, df_indx.EG_intensity_ppp_GDP)
ts.coint(df_indx.GFCF_current, df_indx.CO2_emission)
ts.coint(df_indx.GFCF_current, df_indx['GDP_per_unit_energy_$'])
ts.coint(df_indx.GFCF_current, df_indx.energy_use_per_1000GDP)
ts.coint(df_indx.REC_percent, df_indx.Unemployment_total_percent)
ts.coint(df_indx.REC_percent, df_indx.EG_intensity_ppp_GDP)
ts.coint(df_indx.REC_percent, df_indx.CO2_emission)
ts.coint(df_indx.REC_percent, df_indx['GDP_per_unit_energy_$'])
ts.coint(df_indx.REC_percent, df_indx.energy_use_per_1000GDP)
ts.coint(df_indx.Unemployment_total_percent, df_indx.EG_intensity_ppp_GDP)
ts.coint(df_indx.Unemployment_total_percent, df_indx.CO2_emission)
ts.coint(df_indx.Unemployment_total_percent, df_indx['GDP_per_unit_energy_$'])
ts.coint(df_indx.Unemployment_total_percent, df_indx.energy_use_per_1000GDP)
ts.coint(df_indx.EG_intensity_ppp_GDP, df_indx.CO2_emission)
ts.coint(df_indx.EG_intensity_ppp_GDP, df_indx['GDP_per_unit_energy_$'])
ts.coint(df_indx.EG_intensity_ppp_GDP, df_indx.energy_use_per_1000GDP)
ts.coint(df_indx.CO2_emission, df_indx['GDP_per_unit_energy_$'])
ts.coint(df_indx.CO2_emission, df_indx.energy_use_per_1000GDP)
ts.coint(df_indx['GDP_per_unit_energy_$'], df_indx.energy_use_per_1000GDP)




###################################################################################
#######################    GDP & GFCF (Growth indicators)  ########################
###################################################################################



###################################################
#############     World Analysis ##################
###################################################



##### panel for GFCF world Analysis #####
df_indep=df_indx[['REC_percent','Unemployment_total_percent', 
'EG_intensity_ppp_GDP', 'CO2_emission', 'GDP_per_unit_energy_$'
    ]]




####### GFCF world FirstDifferenceOLS ########
mod_GFCF_world= FirstDifferenceOLS(df_indx.GFCF_current, df_indep)
GFCF_world_Diff = mod_GFCF_world.fit()
print(GFCF_world_Diff)


####### GFCF world PanelOLS #########

PanelOLS(df_indx.GFCF_current, df_indep,entity_effects=True, time_effects=True).fit(debiased=True)






###### GDP_current world FirstDifferenceOLS #####
mod_GDP_world= FirstDifferenceOLS(df_indx.GDP_curr_ppp, df_indep)
GDP_world_Diff = mod_GDP_world.fit()
print(GDP_world_Diff)


##### GDP world panelOLS ####
PanelOLS(df_indx.GDP_curr_ppp, df_indep,entity_effects=True, time_effects=True).fit(debiased=True)




###############################################
#############    Developed   ##################
###############################################


####### Panel for Developed Countries #######
df_developed = df_indx.loc[['Australia', 'Belgium', 'Canada', 'Denmark' , 'France', 'Germany',
'Iceland', 'Japan', 'Luxembourg', 'Netherlands', 'Switzerland', 'United States', 'Sweden'
]]


Developed_indep = df_developed [['REC_percent','Unemployment_total_percent', 
'EG_intensity_ppp_GDP', 'CO2_emission', 'GDP_per_unit_energy_$'
    ]]



####### GFCF developed contries FirstDifferenceOLS ########
mod_GFCF_developed= FirstDifferenceOLS(df_developed.GFCF_current, Developed_indep)
GFCF_developed_Diff = mod_GFCF_developed.fit()
print(GFCF_developed_Diff)


####### GFCF developed Countries PanelOLS #########
PanelOLS(df_developed.GFCF_current, Developed_indep,entity_effects=True, time_effects=True).fit(debiased=True)




###### GDP_current Developed Countries FirstDifferenceOLS #####
mod_GDP_Developed= FirstDifferenceOLS(df_developed.GDP_curr_ppp, Developed_indep)
GDP_developed_Diff = mod_GDP_Developed.fit()
print(GDP_developed_Diff)


##### GDP Developed Developed Countries panelOLS ####
PanelOLS(df_developed.GDP_curr_ppp, Developed_indep
,entity_effects=True, time_effects=True).fit(debiased=True)




###############################################
#############    Developing   #################
###############################################

####### Panel For Developing Countries ######
df_developing = df_indx.loc[['Argentina', 'Chile', 'Chine', 'Egypt, Arab Rep.', 'Israel', 
'Malaysia', 'Mexico', 'Morocco', 'Singapore', 'Turkey', 'Uruguay', 'Russion Dederation']]
 

Developing_indep = df_developing [['REC_percent','Unemployment_total_percent', 
'EG_intensity_ppp_GDP', 'CO2_emission', 'GDP_per_unit_energy_$'
    ]]





####### GFCF developing contries FirstDifferenceOLS ########
mod_GFCF_developing= FirstDifferenceOLS(df_developing.GFCF_current, Developing_indep)
GFCF_developing_Diff = mod_GFCF_developing.fit()
print(GFCF_developing_Diff)


####### GFCF developing Countries PanelOLS #########
PanelOLS(df_developing.GFCF_current, Developing_indep
,entity_effects=True, time_effects=True).fit(debiased=True)



###### GDP_current Developing Countries FirstDifferenceOLS #####
mod_GDP_Developing= FirstDifferenceOLS(df_developing.GDP_curr_ppp, Developing_indep)
GDP_developing_Diff = mod_GDP_Developing.fit()
print(GDP_developing_Diff)


##### GDP world Developing Countries panelOLS ####
PanelOLS(df_developing.GDP_curr_ppp, Developing_indep
,entity_effects=True, time_effects=True).fit(debiased=True)


#################################################################################
##############################    Unemployment   ################################
#################################################################################



#############
### World ###
#############

unemployment_world = df_indx


unemployment_world_indep = unemployment_world [['REC_percent','GFCF_current', 
'EG_intensity_ppp_GDP', 'CO2_emission', 'GDP_per_unit_energy_$'
    ]]

####### Unemployment world contries FirstDifferenceOLS ########
mod_unemployment_world= FirstDifferenceOLS(unemployment_world.Unemployment_total_percent, 
unemployment_world_indep)
unemployment_world_Diff = mod_unemployment_world.fit()
print(unemployment_world_Diff)


####### Unemployment world Countries PanelOLS #########
PanelOLS(unemployment_world.Unemployment_total_percent, unemployment_world_indep
,entity_effects=True, time_effects=True).fit(debiased=True)








#################
### Developed ###
#################

unemployment_developed = df_indx.loc[['Australia', 'Belgium', 'Canada','Denmark' ,'France', 'Germany',
'Iceland', 'Japan', 'Luxembourg', 'Netherlands', 'Switzerland', 'United States', 'Sweden', 'Belguim'
]]


unemployment_Developed_indep = unemployment_developed [['REC_percent','GFCF_current', 
'EG_intensity_ppp_GDP', 'CO2_emission', 'GDP_per_unit_energy_$'
    ]]




####### Unemployment developed contries FirstDifferenceOLS ########
mod_unemployment_developed= FirstDifferenceOLS(unemployment_developed.Unemployment_total_percent, 
unemployment_Developed_indep)
unemployment_developed_Diff = mod_unemployment_developed.fit()
print(unemployment_developed_Diff)


####### Unemployment developed Countries PanelOLS #########
PanelOLS(unemployment_developed.Unemployment_total_percent, unemployment_Developed_indep
,entity_effects=True, time_effects=True).fit(debiased=True)




####################
### Developeding ###
####################


unemployment_developing = df_indx.loc[['Argentina', 'Chile', 'Chine', 'Egypt, Arab Rep.', 'Israel', 
'Malaysia', 'Mexico', 'Morocco', 'Singapore', 'Turkey', 'Uruguay', 'Russion Dederation']]
 

unemployment_Developing_indep = df_developing [['REC_percent','GFCF_current', 
'EG_intensity_ppp_GDP', 'CO2_emission', 'GDP_per_unit_energy_$'
    ]]


####### Unemployment developing contries FirstDifferenceOLS ########
mod_unemployment_developing= FirstDifferenceOLS(unemployment_developing.Unemployment_total_percent, 
unemployment_Developing_indep)
unemployment_developing_Diff = mod_unemployment_developing.fit()
print(unemployment_developing_Diff)


####### Unemployment developing Countries PanelOLS #########
PanelOLS(unemployment_developing.Unemployment_total_percent, unemployment_Developing_indep
,entity_effects=True, time_effects=True).fit(debiased=True)










################################################################################
##################################   CANADA   ##################################
################################################################################

Canada_db = df_indx.loc[['Canada']]

Canada_indep = Canada_db [['REC_percent','Unemployment_total_percent', 
'EG_intensity_ppp_GDP', 'CO2_emission', 'GDP_per_unit_energy_$'
    ]]

Canada_Unemployment_indep = Canada_db [['REC_percent','GFCF_current', 
'EG_intensity_ppp_GDP', 'CO2_emission', 'GDP_per_unit_energy_$'
    ]]

Canada_db.head(30)


####### GFCF Canada FirstDiffeneceOLS ########
mod_Canada_GFCF= FirstDifferenceOLS(Canada_db.GFCF_current, 
Canada_indep)
Canada_GFCF_Diff = mod_Canada_GFCF.fit()
print(Canada_GFCF_Diff)



####### Canada Unemployment   FirstDifferenceOLS ########
mod_Canada_unemployment= FirstDifferenceOLS(Canada_db.Unemployment_total_percent, Canada_Unemployment_indep)
Canada_unemployment_Diff = mod_Canada_unemployment.fit()
print(Canada_unemployment_Diff)


df_indx.unstack()