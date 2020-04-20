# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:16:16 2020

@author: Prasanth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df=pd.read_excel('E:\ml_basics\Concrete_Data.xls')
df.isnull().sum()

df.columns

x=df.copy()
del x['Concrete compressive strength(MPa, megapascals) ']

y=df['Concrete compressive strength(MPa, megapascals) ']


import seaborn as sns



sns.barplot(data=df)
sns.set(rc={'figure.figsize':(11.7,7.27)})


sns.kdeplot(data=x['Cement (component 1)(kg in a m^3 mixture)'])
sns.kdeplot(data=x['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'])
sns.kdeplot(data=x['Fly Ash (component 3)(kg in a m^3 mixture)'])
sns.kdeplot(data=x['Water  (component 4)(kg in a m^3 mixture)'])
sns.kdeplot(data=x['Superplasticizer (component 5)(kg in a m^3 mixture)'])
sns.kdeplot(data=df['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'])
sns.kdeplot(data=df['Fine Aggregate (component 7)(kg in a m^3 mixture)'])


sns.kdeplot(data=x)

sns.lmplot(x='Cement (component 1)(kg in a m^3 mixture)',y='Water  (component 4)(kg in a m^3 mixture)',hue='Age (day)',data=df,palette='coolwarm')
sns.heatmap(data=df.corr(),cmap='coolwarm',annot=True)





plt.boxplot(x['Cement (component 1)(kg in a m^3 mixture)'])
plt.boxplot(x['Blast Furnace Slag (component 2)(kg in a m^3 mixture)']) #ss
per=x['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'].quantile([0,0.998]).values
x['Blast Furnace Slag (component 2)(kg in a m^3 mixture)']=x['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'].clip(per[0],per[1])


plt.boxplot(x['Fly Ash (component 3)(kg in a m^3 mixture)'])


plt.boxplot(x['Water  (component 4)(kg in a m^3 mixture)'])# ss
per=x['Water  (component 4)(kg in a m^3 mixture)'].quantile([0.08,0.985]).values
x['Water  (component 4)(kg in a m^3 mixture)']=x['Water  (component 4)(kg in a m^3 mixture)'].clip(per[0],per[1])


plt.boxplot(x['Superplasticizer (component 5)(kg in a m^3 mixture)'])#ss
per=x['Superplasticizer (component 5)(kg in a m^3 mixture)'].quantile([0,0.985]).values
x['Superplasticizer (component 5)(kg in a m^3 mixture)']=x['Superplasticizer (component 5)(kg in a m^3 mixture)'].clip(per[0],per[1])


plt.boxplot(x['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'])


plt.boxplot(x['Fine Aggregate (component 7)(kg in a m^3 mixture)'])#s
per=x['Fine Aggregate (component 7)(kg in a m^3 mixture)'].quantile([0,0.995]).values
x['Fine Aggregate (component 7)(kg in a m^3 mixture)']=x['Fine Aggregate (component 7)(kg in a m^3 mixture)'].clip(per[0],per[1])


plt.boxplot(x['Age (day)'])#
per=x['Age (day)'].quantile([0,0.94]).values
x['Age (day)']=x['Age (day)'].clip(per[0],per[1])




from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=80)


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(xtrain)


xtrain=scaler.transform(xtrain)
xtest=scaler.transform(xtest)


from sklearn.linear_model import LinearRegression
lreg=LinearRegression()
lreg.fit(xtrain,ytrain)

ypred_ts=lreg.predict(xtest)
ypred_tr=lreg.predict(xtrain)

lreg.coef_
lreg.score

import statsmodels.api as sm
xtrain_sm=sm.add_constant(xtrain)
xtest_sm=sm.add_constant(xtest)

from sklearn.metrics import r2_score
model=sm.OLS(ytrain,xtrain_sm).fit()
predsm=model.predict(xtest_sm)
r2s=r2_score(ytest,predsm)
print(r2s)
print(model.summary())



del x['Coarse Aggregate  (component 6)(kg in a m^3 mixture)']
del x['Fine Aggregate (component 7)(kg in a m^3 mixture)']



import statsmodels.api as sm
xtrain_sm=sm.add_constant(xtrain)
xtest_sm=sm.add_constant(xtest)

from sklearn.metrics import r2_score
model=sm.OLS(ytrain,xtrain_sm).fit()
predsm=model.predict(xtest_sm)
r2s=r2_score(ytest,predsm)
print(r2s)
print(model.summary())

