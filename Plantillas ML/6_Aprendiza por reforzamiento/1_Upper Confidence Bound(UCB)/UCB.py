# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 20:28:09 2021

@author: HP
"""

import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

#Algoritmo de UCB
import math
N = 10000
d = 10
number_of_seletions = [0] * d
sums_of_rewards = [0] * d
ads_select = []
total_reward = 0

for n in range(0,N):
    max_upper_bound = 0
    ad = 0
    for i in range(0,d):
        if(number_of_seletions[i]> 0):
            average_reward = sums_of_rewards[i] / number_of_seletions[i]
            delta_i = math.sqrt(3/2* math.log(n+1) / number_of_seletions[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
            
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_select.append(ad)
    number_of_seletions[ad] = number_of_seletions[ad] + 1
    reward = dataset.values[n , ad]
    sums_of_rewards[ad] = sums_of_rewards[ad]+reward
    total_reward = total_reward + reward
    
#Visualizar Histogramas
plt.hist(ads_select)
plt.title("Histograma de anuncios")
plt.xlabel("ID")
plt.ylabel("Frecuencia de vizualizacion")
plt.show()