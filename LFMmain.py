# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:19:02 2018

@author: yongheng Deng
"""

import random
import LFM
import sys
import os
import math
import imp
imp.reload(LFM)

def loadfile(filename):
    fp=open(filename,'r')

    for i,line in enumerate(fp):
        yield i,line.strip('\r\n')
        if i%100000==0:
            print('loading %s(%s)' % (filename, i),file=sys.stderr)
                
    fp.close()
    print('load %s succ' % filename,file=sys.stderr)

def generate_dataset(filename,pivot=0.7):
    trainset_len = 0
    testset_len = 0
    count = 0
    trainset={}
    testset={}
    
    for i,line in loadfile(filename):
        if i==count:
            user,rated=line.split('|')
            count=count+int(rated)+1
            continue
        else:
            item,rating=line.split()
        #user, movie, rating, _ = line.split('::')

        if random.random()<pivot:
            trainset.setdefault(user,{})
            trainset[user][item]=rating
            trainset_len+=1
        else:
            testset.setdefault(user,{})
            testset[user][item]=rating
            testset_len+=1

    print('split training set and test set succ',file=sys.stderr)
    print('train set = %s' % trainset_len,file=sys.stderr)
    print('test set = %s' % testset_len,file=sys.stderr)
    
    return trainset,testset

def generate_resultset(filename,P,Q):
    fp=open('data-new/result.txt','w')
    resultset={}
    count = 0
    for i,line in loadfile(filename):
        if i==count:
            fp.writelines(line+'\n')
            user,rated=line.split('|')
            resultset.setdefault(user,{})
            count=count+int(rated)+1
            continue
        else:
            item=line
            resultset[user][item]=LFM.Predict(user,item)
            fp.writelines(line+' '+str(int(round(resultset[user][item])))+'\n')
        #user, movie, rating, _ = line.split('::')
        
    fp.close()
    return resultset
            
if __name__ == '__main__':
    ratingfile=os.path.join('data-new','train.txt')
    testfile=os.path.join('data-new','test.txt')
    train,test=generate_dataset(ratingfile)
#    P,Q=LFM.InitModel(train,10)
#    print(P['0'][0])
P,Q = LFM.LatentFactorModel(train, 10,20, 0.000001, 0.01)
#copute the RMSE
rmse=0.00
count=0
for user, items in test.items():
    for item,rating in items.items():
        predict_rating=LFM.Predict(user,item)
        rmse+=(int(rating)-predict_rating)*(int(rating)-predict_rating)
        count+=1
rmse=math.sqrt(rmse/count)
print('the rmse=',rmse)
result=generate_resultset(testfile,P,Q)
    