#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[98]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[99]:


get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[100]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[101]:


# Sua análise da parte 1 começa aqui.


# In[102]:


dataframe.head()


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[103]:


round(np.quantile(dataframe['normal'], 0.5),3)


# In[104]:


round(np.quantile(dataframe['normal'], 0.25),3)


# In[105]:


round(np.quantile(dataframe['normal'], 0.75),3)


# In[106]:


dataframe.quantile(.5)


# In[107]:


dataframe.quantile(.25)


# In[108]:


dataframe.quantile(.75)


# In[109]:


round(np.quantile(dataframe['normal'], 0.5),3)


# In[110]:


def q1():
    a = round(np.quantile(dataframe['normal'], 0.25) - np.quantile(dataframe['binomial'], 0.25),3)
    b = round(np.quantile(dataframe['normal'], 0.5)  - np.quantile(dataframe['binomial'], 0.5),3)
    c = round(np.quantile(dataframe['normal'], 0.75) - np.quantile(dataframe['binomial'], 0.75),3)
    return(a,b,c)


# In[111]:


print(q1())


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[112]:


def q2():
    
    xS = dataframe['normal'].mean()-dataframe['normal'].std()
    xS2 = dataframe['normal'].mean()+dataframe['normal'].std()
    pInterval = sct.norm.cdf([xS,xS2] , loc = dataframe['normal'].mean(), scale = dataframe['normal'].std())
    
    
    return round(pInterval[1]-pInterval[0],3)


# In[113]:


print(q2())


# ### intervalo:  $[\bar{x} - 2s, \bar{x} + 2s]$ 

# In[114]:


xS = dataframe['normal'].mean()-2*dataframe['normal'].std()
xS2 = dataframe['normal'].mean()+2*dataframe['normal'].std()
pInterval = sct.norm.cdf([xS,xS2] , loc = dataframe['normal'].mean(), scale = dataframe['normal'].std())
round(pInterval[1]-pInterval[0],3)


# ### Intervalo: $[\bar{x} - 3s, \bar{x} + 3s]$.

# In[115]:


xS = dataframe['normal'].mean()-3*dataframe['normal'].std()
xS2 = dataframe['normal'].mean()+3*dataframe['normal'].std()
pInterval = sct.norm.cdf([xS,xS2] , loc = dataframe['normal'].mean(), scale = dataframe['normal'].std())
round(pInterval[1]-pInterval[0],3)


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[116]:


dataframe.var()


# In[117]:


dataframe.mean()


# In[118]:


def q3():
    
    difMean = dataframe['binomial'].mean() - dataframe['normal'].mean()
    difVar =  dataframe['binomial'].var() - dataframe['normal'].var()
    return(round(difMean,3),round(difVar,3))
    


# In[119]:


print(q3())


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# In[120]:


# provavelmente vao se aproximar até serem praticamente o mesmo


# ## Parte 2

# ### _Setup_ da parte 2

# In[121]:


stars = pd.read_csv("HTRU2/HTRU_2.csv")


# In[122]:



stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[123]:


stars.head(2)


# In[124]:


# Sua análise da parte 2 começa aqui.


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[125]:


temp = stars[stars['target']==False]['mean_profile']
false_pulsar_mean_profile_standardized = stars[stars['target']==False]['mean_profile'].apply(lambda x: (x-temp.mean())/temp.std()) 


# In[126]:


false_pulsar_mean_profile_standardized.std()


# In[127]:


def q4():
    
    temp = stars[stars['target']==False]['mean_profile']
    false_pulsar_mean_profile_standardized = stars[stars['target']==False]['mean_profile'].apply(lambda x: (x-temp.mean())/temp.std()) 
    
    media = false_pulsar_mean_profile_standardized.mean()
    std = false_pulsar_mean_profile_standardized.std()
    
    a = round(sct.norm.ppf(0.80, loc=media, scale=std),3)
    b = round(sct.norm.ppf(0.90, loc=media, scale=std),3)
    c = round(sct.norm.ppf(0.95, loc=media, scale=std),3)
    
    return (a,b,c)


# In[128]:


print(q4())


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[129]:


q1 = false_pulsar_mean_profile_standardized.quantile(.25)
q2 = false_pulsar_mean_profile_standardized.quantile(.50)
q3 = false_pulsar_mean_profile_standardized.quantile(.75)
print (q1,q2,q3)


# In[130]:


q1Norm = dataframe['normal'].apply(lambda x: (x-dataframe['normal'].mean())/dataframe['normal'].std()).quantile(.25)
q2Norm = dataframe['normal'].apply(lambda x: (x-dataframe['normal'].mean())/dataframe['normal'].std()).quantile(.50)
q3Norm = dataframe['normal'].apply(lambda x: (x-dataframe['normal'].mean())/dataframe['normal'].std()).quantile(.75)
print(q1Norm,q2Norm,q3Norm)


# In[131]:


def q5():
    
    q1 = false_pulsar_mean_profile_standardized.quantile(.25)
    q2 = false_pulsar_mean_profile_standardized.quantile(.50)
    q3 = false_pulsar_mean_profile_standardized.quantile(.75)
    
    q1Norm = dataframe['normal'].apply(lambda x: (x-dataframe['normal'].mean())/dataframe['normal'].std()).quantile(.25)
    q2Norm = dataframe['normal'].apply(lambda x: (x-dataframe['normal'].mean())/dataframe['normal'].std()).quantile(.50)
    q3Norm = dataframe['normal'].apply(lambda x: (x-dataframe['normal'].mean())/dataframe['normal'].std()).quantile(.75)
  
    return (round((q1Norm - q1),3), round((q2Norm - q2),3), round((q3Norm - q3),3))


# In[132]:


print(q5())


# ## Resumo interessante: Estralas que nao sao pulsarem tem um perfil medio de distribuicao praticamente normal. Provavelmente os pulsares nao.

# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
