# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:24:21 2024

@author: User
"""
#%%Instalando os pacotes
!pip install pingouin

#%%Importando os pacotes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pingouin as pg
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

#%% Importando banco de dados

dados_ecommerce = pd.read_csv('data.csv', encoding= 'unicode_escape', sep=',')

#%% Entendendo os dados

# Informações gerais das colunas e tipo das variáveis
dados_ecommerce.info()

# Estatistica descritiva das variáveis
tab_desc = dados_ecommerce.describe()

# Somando total de valores nulos 

porc_nulos_costumerID = (dados_ecommerce['CustomerID'].isna().sum() / len(dados_ecommerce['CustomerID'])) * 100
porc_nulos_description = (dados_ecommerce['Description'].isna().sum() / len(dados_ecommerce['CustomerID'])) * 100
print('{:.2f}%, {:.2f}%'.format(porc_nulos_costumerID,porc_nulos_description))

# - Missing em CostumerID (24,93%) e Description (0,27%)
# - Valores negativos em Quantity e UnitPrice
# - InvoiceDate deveria ser tipo date

#%% Preparação dos dados e limpeza

# Removendo os valores missings

dados_ecommerce.drop(dados_ecommerce[dados_ecommerce['CustomerID'].isna() | dados_ecommerce['Description'].isna()].index, inplace=True)
dados_ecommerce.info()

# Removendo valores de Quantity e UnitPrice negativos

dados_ecommerce.drop(dados_ecommerce[dados_ecommerce['Quantity']<0].index, inplace=True)
dados_ecommerce.describe()

# Ajustando tipos de dados errados

dados_ecommerce['InvoiceDate']=pd.to_datetime(dados_ecommerce['InvoiceDate'])

dados_ecommerce['CustomerID'] = dados_ecommerce['CustomerID'].astype('int')

dados_ecommerce.info()

#%% Calculando Frequency

dados_ecommerce['Rank'] = dados_ecommerce.sort_values(['CustomerID','InvoiceDate']).groupby(['CustomerID'])['InvoiceDate'].rank(method='min').astype(int)
recent_df = dados_ecommerce[dados_ecommerce['Rank']==1]

earliest_purchase = pd.to_datetime(min(recent_df['InvoiceDate']))
recent_df['Recency'] = (recent_df['InvoiceDate'] - earliest_purchase).dt.days
#RFM porque nos permite calcular a recência com base na primeira interação de cada cliente com a empresa. Isso proporciona uma medida precisa 
#de quanto tempo se passou desde a primeira compra de cada cliente até a data de referência atual, 
#o que é essencial para entender e segmentar clientes com base em seu comportamento de compra recente.