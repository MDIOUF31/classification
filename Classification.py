#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
bd=pd.read_csv("African_crises_dataset.csv")
bd


# In[2]:


#Informations générales sur l'ensemble de données
bd.info()

Le taux d’inflation annuel de l’IPC)La base de données qui est soumise à notre étude se concentre sur les crises bancaires, de la dette, financière, inflationniste et systémique qui se sont produites, de 1860 à 2014, dans 13 pays africains. Elle comporte quatorze variables dont trois variable catégorielles que sont:le code du pays(country_code),le pays(country) et la situation de crise bancaire(banking_crisis). Les onze autres variables sont des variables numériques et sont:le numéro du pays(country_number),l'année(year),systemic_crisis(cette colonne prend soit 0 ou 1),exch_usd(Le taux de change du pays par rapport au dollar américain),domestic_debt_in_default("0" signifie qu'aucun défaut de dette souveraine intérieure n'a eu lieu au cours de l'année et "1" signifie qu'un défaut de dette souveraine intérieur a eu lieu au cours de l'année),sovereign_external_debt_default(« 0 » signifie qu'aucun défaut de dette souveraine extérieure n'a eu lieu au cours de l'année et « 1 » signifie qu'un défaut de dette souveraine extérieur a eu lieu au cours de l'année), gdp_weighted_default(La dette totale en défaut par rapport au PIB),inflation_annual_cpi(Le taux d’inflation annuel de l’IPC)et les trois variables restantes(independence,currency_crises,inflation_crises) prennent soit zero ou 1.
Au total, la base de données est constituée de 1058 observations et ne comporte pas de valeurs manquantes en son sein.
# In[3]:


#Les pays concernés par l'observation
bd["country"].unique()


# In[4]:


#Aperçu statistique sur la base de données
bd.describe()


# In[5]:


get_ipython().system('pip install ydata_profiling')


# In[6]:


#Créez des rapports de profilage de pandas pour obtenir des informations sur l'ensemble de données
from ydata_profiling import ProfileReport
profile=ProfileReport(bd,title="Rapport de l'ensemble de données concernant la Crise systémique, crise bancaire, crise d'inflation en Afrique ")
profile.to_file("Rapport de l'ensemble de données concernant la Crise systémique, crise bancaire, crise d'inflation en Afrique.html")

L'analyse du rapport concernant les crises bancaires, de la dette, financière, inflationniste et systémique qui se sont produites, de 1860 à 2014, dans 13 pays africains révéle une corrélation faible voir même nulle pour la plus part des variables. Cependant,on constate une corrélation moyenne et positive entre l'année(year) et le taux de change du pays par rapport au dollar américain(exch_usd) d'une valeur de 0,512. On note également que la variable "inflation_annual_cpi" présente une forte asymétrie positive.
# In[7]:


import plotly.express as px
fig=px.scatter(bd,x="year",y="banking_crisis",color="country",title="la situation de crise ou pas de chaque pays par année")
fig.show()

La figure ci-dessus traduit la situation de crise ou pas de chaque pays par année.Une étude du graphe révéle que les situations de crise sont plus courantes à partir de l'année 1976 et qu'elles sont plus présentes en Zimbabwe.
# In[8]:


fig=px.bar(bd,x="year",y="exch_usd",color="country",
           title="Le taux de change de chaque pays par rapport au dollar américain par année")
fig.show()

Le graphique ci-dessus traduit l'évolution du taux de change de chaque pays par rapport au dollar américain(exch_usd) par année pendant la période de 1940 à 2014.
# In[9]:


fig=px.histogram(bd,x="year",y="inflation_annual_cpi",color="country",
                 title="Evolution de l'inflation_annual_cpi par année dans chaque pays")
fig.show()

Comme son nom l'indique, le graphe traduit l'évolution de l'inflation_annual_cpi par année dans chaque pays. L'inflation_annual_cpi est le taux d'inflation annuel de l'indice des prix à la consommation(IPC). En effet, l'IPC est l'instrument de mesure de l'inflation. Il permet d'estimer la variation moyenne des prix des produits consommés par les ménages entre deux périodes. Dans notre étude du graphe, nous nous concentrerons principalement sur les valeurs de déflation négatives (c'est à dire inférieur à 0) afin de mettre en exergue les baisses des prix les plus importantes dans les pays concernés. Cependant un Coup d'œil rapide sur le graphe vous permet de constater des déflations positives. Ces dernières sont aussi des baisses des prix à la consommation mais elles ne sont pas si importantes comme celles dont nous montrerons dans les lignes qui suivent.
L'analyse du graphe montre une déflation de l'IPC durant la période de 1875 à 1884 pour l'Algérie et l'Egypte. De 1885 à 1889, on observe une déflation importante de l'IPC en Egypte. En Afrique du SUD, on constate une déflation de l'IPC de 1905 à 1909. De 1920 à 1924,on note une déflation de l'IPC en Egypte,en Afrique du SUD et en Angola. La période de 1930 à 1934 est marquée par une déflation en Zimbabwe, en Egypte, en Afrique du SUD et en Angola. En Egypte, on remarque également une déflation de 1945 à 1949. La dernière déflation dans cette observation a été remarquée en Angola de 1950 à 1954.
Il faut cependant noter l'inflation importante en Zimbabwe de 2005 à 2009 d'une valeur de 22,05783M car celà traduit une cherté  extrême de la vie.
# In[10]:


#import des bibliothéques nécessaires
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns


# In[11]:


#conversion de la variable catégorielle (banking_crisis) en variable numérique
bd["banking_crisis"]=bd["banking_crisis"].map({"crisis": 1, "no_crisis": 0})   
bd.head()


# In[12]:


#choix des features et du target
y = bd["banking_crisis"]
features=["year","systemic_crisis","exch_usd","inflation_annual_cpi","independence",
        "currency_crises","inflation_crises"]
x=bd[features]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=35)


# In[13]:


logreg = LogisticRegression()   #build our logistic model
logreg.fit(x_train, y_train)  #fitting training data
y_pred  = logreg.predict(x_test)    #testing model’s performance
print("précision={:.2f}".format(logreg.score(x_test, y_test)))


# In[14]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[15]:


confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)

