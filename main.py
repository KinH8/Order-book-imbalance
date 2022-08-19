# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 23:24:35 2022

@author: Wu
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Gould and Bonart (2015) Queue Imbalance as a one-tick ahead price predifctor in a limit order book

df = lambda x: datetime.utcfromtimestamp(float(x) // 1000000000) + timedelta(hours=8)

data=pd.read_csv('ModelDepthProto_20170224.csv', index_col='timestampNano', date_parser=df)
data = data.loc[data.index[0]+ timedelta(minutes=30):data.index[0]+ timedelta(hours=6.5)]

data['mid'] = (data['bid1p']+data['ask1p'])/2
data['nb'] = data['bid1q']+data['bid2q']+data['bid3q']+data['bid4q']+data['bid5q']
data['na'] = data['ask1q']+data['ask2q']+data['ask3q']+data['ask4q']+data['ask5q']
data['I'] = (data['nb']-data['na'])/(data['nb']+data['na'])
data['y'] = data['mid'].pct_change()
data['y'] = data['y'].mask(data['y']<0,0).mask(data['y']>0,1).mask(data['y']==0,np.nan)

# Estimated I
data['Is'] = np.nan

z = sorted(pd.Index(data['y']).get_indexer_for([0,1]))

for x,y in enumerate(z):
    if x == 0:
        temp = data.iloc[:y]
    else:
        temp = data.iloc[z[x-1]:y]
    
    #if len(temp)>1:
    data['Is'].iloc[y] = temp.sample(n=1)['I'].values[0]

data['Is'].hist()
plt.title('Distribution of orderbook imbalance')

cutoff=1/4
res = data[['y','Is']].dropna()
#res = res[(res['Is']>cutoff)|(res['Is']<-cutoff)]  # Uncomment for conditional Is
#plt.scatter(x=res['y'],y=res['Is'])

train = res.iloc[:int(len(res)*0.8)]
test = res.iloc[-int(0.2*len(res)):]

clf = LogisticRegression(random_state=0).fit(train['Is'].values.reshape(-1, 1), train['y'])
print('Prediction: ', clf.predict(test['Is'].values.reshape(-1, 1)))
print('Prediction probability: ', clf.predict_proba(test['Is'].values.reshape(-1, 1)))
print('Score: ',clf.score(test['Is'].values.reshape(-1, 1),test['y']))