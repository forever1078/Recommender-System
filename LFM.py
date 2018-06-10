# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:51:55 2018

@author: yongheng Deng
"""

import random
import math
import numpy as np

P={}
Q={}

def Predict(u, i):
    ret=0
    for f in range(0,len(P[u])):
        if i not in Q.keys():
            return 50.0
        else:
            ret+=(P[u][f]) * (Q[i][f])
        #print(p[u][f],q[i][f],ret)
    if ret<0:
        ret=0
    if ret>100:
        ret=100
    return ret

def InitModel(user_items,F):
#    P = dict()
#    Q = dict()
    for user, items in user_items.items():
        P[user] = dict()
        for f in range(0,F):
            P[user][f] = random.random()*20/math.sqrt(F)
            #print(P[user][f])
        for i,r in items.items():
            if i not in Q:
                Q[i] = dict()
                for f in range(0,F):
                    Q[i][f] = random.random()*20/math.sqrt(F)
                    #print(Q[i][f])
    return P,Q

def LatentFactorModel(user_items, F,T, alpha, lamb):
 #   InitAllItemSet(user_items)
    InitModel(user_items, F)
    for step in range(0,T):
        for user, items in user_items.items():
            #samples = RandSelectNegativeSample(items)
            #for item, rui in samples.items():
            for item,rui in items.items():
                ui=Predict(user, item)
                eui = int(rui) - ui
                #print(int(rui),ui,eui)
                for f in range(0,F):
                    P[user][f] += alpha * (eui * Q[item][f] - \
                        lamb * P[user][f])
                    Q[item][f] += alpha * (eui * P[user][f] - \
                        lamb * Q[item][f])
                #if abs(eui)<10:
                    #return P,Q
        alpha *= 0.9
#    plt.plot(loss)
    return P,Q
        