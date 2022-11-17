#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import matplotlib.ticker as mtick
import os


def evaluate_arima_model(X,order):
    # prepare training dataset
    results={}
    train_size = int(len(X) * 0.680)
    train, test = X[0:train_size], X[train_size:]
    tr = train.tolist()
    te = test.tolist()
    history = [x for x in tr]
    #return(test[0])
    # make predictions
    predictions = list()
#    for t in range(len(test)):
#        model = ARIMA(history, order=order)
#        model_fit = model.fit()
#        yhat = model_fit.forecast()[t]
#        predictions.append(yhat)
    for t in range(len(te)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(te[t])

        #history.append(test[t])
    # calculate out of sample error
    #return(history)
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    results['rmse']=rmse
    results['mse']=mse
    t = test.tolist()
    results['test']=t
    results['predictions']=predictions
    results['df_full']=history
    
    return(results)

#print(evaluate_arima_model(df3['latepercentage'].squeeze(),(1,1,1)))



# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    #print((order,mse))
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    #print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    #print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return(best_cfg, best_score)


# evaluate parameters
#p_values = [0, 1, 2, 4, 6, 8, 10]
#d_values = range(0, 3)
#q_values = range(0, 3)
warnings.filterwarnings("ignore")


# In[6]:


def forecasting(df,monthstopredict,lod):
    vendors={522182: (0, 1, 0), 503894: (10, 1, 1), 522865: (2, 1, 1), 556264: (1, 0, 2), 569547: (0, 1, 0), 503976: (0, 1, 1), 542319: (2, 0, 2), 501967: (4, 0, 2), 529485: (4, 0, 2), 567926: (0, 0, 0), 558725: (2, 1, 1), 557704: (0, 2, 1), 566041: (0, 1, 2), 551948: (2, 0, 2), 500288: (4, 1, 0), 503769: (10, 0, 0), 559749: (2, 0, 1), 505439: (2, 1, 0), 569836: (0, 1, 0), 576663: (6, 0, 2), 500003: (0, 1, 0), 539850: (1, 1, 2), 521999: (0, 2, 2), 516114: (0, 0, 0), 531182: (1, 0, 0), 554748: (0, 0, 0), 543952: (1, 1, 2), 519637: (4, 1, 1), 505993: (4, 2, 1), 560313: (0, 1, 0), 564891: (0, 1, 0), 569218: (0, 1, 0), 519227: (0, 1, 0), 574279: (0, 1, 0), 533389: (0, 1, 0), 578425: (0, 0, 0), 562185: (0, 2, 0), 513478: (0, 1, 0), 500487: (0, 1, 0), 533876: (0, 1, 0), 563036: (0, 0, 0), 565088: (1, 2, 2), 550906: (0, 1, 0)}
    materiels={'X714': (0, 1, 1), '057': (0, 2, 2), '013': (2, 1, 1), '073': (0, 0, 0), '014': (0, 0, 1), '060': (1, 1, 1), '018': (0, 1, 0), '012': (0, 1, 1), '030': (0, 2, 1), '017': (1, 1, 0), '041': (0, 0, 1), '048': (2, 1, 2), '032': (2, 0, 0), '040': (1, 0, 0), '021': (6, 0, 2), 'X711': (4, 1, 2), '009': (1, 2, 2), '059': (2, 0, 2), '002': (0, 0, 0), '047': (0, 0, 0), '027': (0, 1, 0), '061': (0, 1, 0), '011': (0, 1, 0)}
    vendors_materiels={'500061X721': (0, 1, 0), '500288048': (4, 1, 0), '500487073': (0, 1, 0), '501967057': (2, 1, 2), '501967060': (2, 1, 0), '503769032': (10, 0, 0), '503894021': (0, 0, 0), '505439017': (2, 1, 0), '505993073': (4, 2, 1), '512330X709': (0, 1, 0), '513478061': (0, 1, 0), '516114060': (0, 0, 0), '519227057': (0, 1, 0), '521999012': (0, 2, 2), '522182X714': (0, 1, 0), '522865013': (2, 1, 1), '527821040': (0, 0, 0), '529485018': (0, 1, 0), '529485041': (0, 0, 0), '530692X711': (0, 1, 0), '533876057': (0, 1, 0), '539850008': (0, 0, 0), '539850009': (0, 1, 1), '542319014': (0, 0, 0), '542319059': (10, 1, 1), '543952011': (0, 1, 0), '543952009': (1, 1, 2), '550906X714': (0, 1, 0), '551948073': (2, 0, 2), '554748002': (0, 0, 0), '556264027': (0, 1, 0), '556264073': (0, 0, 0), '557704030': (0, 2, 1), '558725012': (2, 1, 1), '559749040': (0, 1, 1), '560313060': (0, 1, 0), '562185009': (0, 2, 0), '563036056': (0, 0, 0), '564891X711': (0, 1, 0), '565088001': (10, 2, 2), '566041017': (0, 1, 2), '567926060': (0, 0, 0), '569218059': (0, 1, 0), '569547013': (10, 1, 1), '569836060': (0, 0, 1), '574279040': (0, 1, 0), '576663032': (6, 0, 2), '578425017': (0, 0, 0)}
    mats={'920107304': (0, 2, 0), '1027949001-6295': (0, 0, 0), '930201037': (0, 0, 0), 'MRO\xa0195803': (0, 1, 0), '1071767001': (0, 0, 0), '920107120': (0, 0, 0), '1071601001-6205': (0, 0, 0), '1025350001': (0, 1, 0), '1025346001-9160': (4, 0, 1), '1036670001': (1, 2, 1), '920107301': (0, 2, 2), '1072524001': (0, 1, 0), '191445005': (0, 1, 0), '1027357001': (0, 1, 0), '465C4901': (0, 1, 0), '1027948001-6205': (1, 0, 2), '1027952001-6205': (0, 0, 0), '1025347001': (0, 1, 0), '1026958001-6205': (1, 2, 1), '1025207001-6205': (2, 1, 0), '865246501': (0, 1, 0), '1072524002': (0, 1, 0), '1025348001': (0, 1, 0), '209774019': (0, 1, 0), '1038038001-7360': (10, 0, 2), '1025207002-6205': (2, 1, 0), '1037811005': (4, 1, 2), '1027949001-6205': (0, 0, 1), '1068328001': (2, 1, 0), '1064752001': (0, 0, 0), '1025349001': (0, 1, 0), '1234790001': (1, 0, 0), '1101030001': (2, 2, 0), '1027352001': (2, 2, 2), '1038047001': (2, 1, 0), '1027948001-6053': (2, 2, 2), '10263040010001': (0, 1, 0), '1126459001': (0, 1, 0), '1027350001': (0, 0, 0), '1025207001-6053': (2, 1, 0), '1027350002': (1, 0, 0), '1071601001-6249': (4, 1, 1), '1027949001-6249': (0, 0, 0), '1071601001-6295': (2, 1, 1), '1026305002': (1, 0, 0), '1025207002-6053': (1, 0, 2), '1027351001-6053': (1, 0, 0), '442F0204': (4, 0, 0), 'V_00036914-001': (0, 1, 0), '1026958001-6053': (0, 1, 0), '1026286001': (1, 0, 1), '1027349001': (2, 0, 2), '1027351001-6205': (1, 0, 0), '442H4901': (0, 1, 0), '1027962001-6205': (4, 2, 2), '1027356001-6205': (2, 1, 0), '930201043': (0, 1, 0), '230415004': (6, 0, 1), '930200064': (0, 0, 0), '920107121': (1, 2, 2), '209774023': (0, 1, 0), '920107300': (1, 1, 1), '1027962001-6053': (0, 0, 2), '1038033002-6205': (0, 1, 0), '10263040010004': (2, 0, 2), '858000252': (0, 1, 0), '1026305001': (0, 0, 0), '1038033001-6205': (0, 1, 0), '1025484001': (0, 0, 0), '1085708001': (0, 0, 0), '1025286001': (0, 0, 0), '1025288001': (0, 0, 0), '1038038001-4140': (6, 0, 2), '1025487001': (0, 0, 0), '1023934001': (0, 0, 0), '1038011001': (0, 0, 0), '230415005': (0, 1, 0), '1023931001': (1, 1, 2), '1025486001': (0, 0, 0), '209774024': (6, 0, 1), '1038012001': (1, 0, 0), '1023933001': (0, 1, 0), '1038018001': (0, 0, 0), '1023933002': (0, 1, 0), '1025287001-9180': (0, 2, 2), '1023930001': (1, 1, 0), '1025483001': (1, 1, 0), '1025485001': (0, 0, 2), '1025287002-9180': (6, 0, 1), '906600050-L750': (0, 1, 0), '1038017001': (0, 0, 0), '1024885001': (0, 0, 0), '1038020001': (0, 0, 1), 'MRO\xa0109786': (0, 1, 0), '1023937001-9160': (0, 1, 2), 'Y881547400': (0, 1, 0), '1038033001-6059': (0, 1, 0), 'V_00034501-24': (0, 1, 0), '1036430001-5T29': (0, 1, 0), '1217129001': (0, 1, 0), '1023929001': (0, 0, 0), 'V_00030593-403': (0, 1, 0), 'V_00030593-603': (0, 1, 0), '906600050-L112': (4, 1, 1), 'MRO\xa0190122': (0, 1, 0), 'V_00027061-503': (0, 2, 1), 'V_00032761-591': (0, 1, 0), 'V_00026383-701': (0, 1, 0), 'V_00017190-802': (0, 1, 0), '465E3101': (0, 1, 0), 'V_00033455-803': (0, 1, 0), '1333689001': (0, 1, 0), '1331891001': (0, 0, 2), 'V_000390811': (0, 1, 0), 'MRO\xa0190060': (0, 1, 0), '930201045': (6, 2, 0), 'MRO\xa0190155': (0, 1, 0), 'MRO\xa0190159': (0, 1, 0), 'MRO\xa0190154': (0, 1, 0), 'V_00032961-403': (6, 1, 0), '920107306': (0, 0, 0), 'V_00034826-5': (0, 1, 0), 'V_000379248': (0, 1, 0), 'V_00012504-731': (0, 1, 0), 'V_00026527-701': (0, 1, 0), '920107307': (0, 1, 0), 'V0003757610': (0, 1, 0), 'V_0003736027': (0, 1, 0), '929100169': (0, 1, 0), 'V_00032961-103': (0, 1, 0), '1038034001-6205': (0, 2, 2), '1038034002-6205': (0, 2, 2), 'V_00029491-803': (0, 1, 0), 'V_00029492-601': (0, 1, 0), 'V_00037063-401': (0, 1, 0), 'V_00036620-8': (0, 1, 0), 'V_00035801-403': (0, 1, 0), 'MRO\xa0190065': (0, 1, 0), 'V_00027146-684': (0, 1, 0), '1356946001-6205': (2, 0, 2), 'V_00033353-301': (0, 1, 0), 'V_00033944-3': (0, 1, 0), 'V_00035823-201': (0, 1, 0), 'V_00029492-403': (0, 1, 0), 'V_00012504-170': (0, 1, 0), 'V_00026592-15': (0, 1, 0), 'V00037845403': (0, 1, 0), '906600050-L767': (0, 1, 0), 'MRO\xa0140829': (0, 1, 0), 'V_00034835-29': (0, 1, 0), 'V_00017436-801': (2, 1, 1), '1406644001': (0, 1, 0), 'V00037574902': (0, 1, 0), 'MRO\xa0153349': (0, 1, 0), 'V_00036539803': (0, 1, 0), 'V_000382251': (0, 1, 0), 'V_00024544-17': (0, 1, 0), '920300104': (0, 0, 0), 'V_00034231-21': (0, 1, 0), '920300113': (0, 1, 0), 'V_00029492-901': (0, 1, 0), '920107311': (0, 1, 2), 'V_00026352-805': (0, 1, 0), 'V000380914': (0, 1, 0), 'V_0003803212': (0, 1, 0), 'V_00015704-9': (0, 1, 0), 'V_00037541801': (0, 1, 0), 'V_00026352704': (0, 1, 0), 'V_00035552-11': (0, 1, 0), '1426978001': (1, 0, 2), 'V_00036452-202': (0, 1, 0), 'V_00026352814': (0, 1, 0), 'V_00034942-66': (0, 1, 0), 'MRO\xa0195839': (0, 1, 0), 'V_00035552S-11': (0, 0, 0), 'V_00038370404': (0, 1, 0), 'V_00033956-802': (0, 1, 0), '1062714003': (0, 1, 0), 'V_00033941701': (0, 1, 0), 'V_00036483-17': (0, 1, 0), '1303896001': (0, 1, 0), '1392929002': (0, 1, 0), '895048256-6205': (0, 1, 0), '1392928002': (0, 1, 0), '1392929001': (0, 1, 0), '895048259-6205': (0, 0, 0), 'V_00027674-3': (0, 1, 0), '920107312': (0, 0, 0), '920107313': (0, 1, 1), 'V_00032961-802': (10, 1, 1), 'V_00035801-802': (0, 1, 0), 'MRO\xa0131960': (0, 1, 0), '1442753001': (0, 0, 0), 'V_00036226-37': (0, 1, 0), '91038': (0, 1, 0), '920300618': (0, 0, 1), 'V_00028164-12': (0, 1, 0), '1303890001-6205': (0, 1, 2), '1303891001-6059': (0, 1, 0), '209774025': (0, 1, 0), 'V_00036219-403': (0, 1, 0), 'V_00011841-001': (0, 1, 0), '1388410001-6527': (0, 1, 0), '1075081001-6205': (0, 1, 0), 'SPL92469C01': (0, 1, 0), '1388410001-6205': (0, 1, 0), 'MRO\xa0102953': (0, 1, 0), 'V_0004052319': (0, 1, 0), 'V_0003687120': (0, 1, 0), 'V_00028491-4': (0, 1, 0), 'V_00029103-401': (0, 1, 0), '1303891001-6205': (0, 1, 0), '1303890001-6059': (0, 1, 0), 'V_00028491S-4': (0, 1, 0), '1267709001': (0, 1, 0), '1303891001-6053': (0, 1, 0), 'V_00034942-36': (4, 1, 0), '1352152001-6059': (0, 1, 0), '1354659001-6205': (0, 1, 0), '1352135001': (0, 1, 0), '1352099001-6527': (0, 1, 0), '1352152002-6059': (0, 1, 0), '1352139001-6205': (0, 0, 0), '1352139002-6059': (4, 1, 1), 'V_00032953-801': (0, 1, 0), '1352139002-6205': (0, 0, 0), '1352134001-6205': (4, 1, 1), '1352139001-6059': (4, 1, 1), '1445156001KR07': (0, 1, 0), 'V_00033855-805': (0, 1, 0), 'V000393210780': (0, 1, 0), 'AIR33700': (0, 1, 0), 'V_00015410-803': (0, 1, 0), '1446876001': (0, 1, 0), '1446873001': (0, 1, 0), '1446877001': (0, 1, 2), '1446874001': (10, 2, 2), '1446878001': (0, 2, 0), '1446878002': (0, 2, 0), '1446879001': (0, 2, 0), 'V_000405363': (0, 1, 0), 'V_00023102-601': (0, 1, 0), 'V_00030341-48': (0, 1, 0), 'MRO\xa0190042': (0, 1, 0), '1455267001': (0, 1, 0), '100459025': (0, 1, 0), '1267710001': (0, 0, 0), '1267710002': (0, 0, 0), '920106669': (1, 1, 2), '920106670': (0, 0, 0), 'V_00032143-802': (0, 1, 0), 'V_00032143-201': (0, 1, 0), '1446881001': (0, 1, 0), '91037': (0, 1, 0), 'V_000407741': (0, 1, 0), '1472244001': (0, 1, 0), '1173489001': (0, 1, 0), 'V_00019435-4': (0, 1, 0), '1467598004': (0, 1, 0), 'V_00041436403': (0, 0, 0), '1467596001': (1, 2, 2), '1467598002': (0, 0, 0), 'MRO\xa0101174': (0, 1, 0), 'V_000284603': (0, 1, 0), 'V_00028460-2': (0, 1, 0), 'V_00034942-6': (0, 0, 0), '1352134001-6059': (2, 0, 0), '1352134002-6059': (2, 0, 0), '1352134002-6205': (0, 1, 0), 'V_00032961-102': (0, 1, 0), 'V_00033832-001': (0, 1, 0), 'V000425676': (0, 1, 0), 'V_00034942-12': (0, 1, 0), 'SPL5J08': (0, 0, 0), 'V_0004112919': (0, 1, 0), 'V_00034942-20': (0, 1, 0), 'V_00034942-70': (0, 1, 0)}
    full_df = (1,0,0)
    if lod == 'Supplier and Materiel Group':
        vendor = df['Vendor_number'].iloc[0]
        matgroup = df['Material_Group'].iloc[0]
        fullname = str(vendor)+matgroup
        order = vendors_materiels[fullname]
    elif lod == 'Supplier':
        vendor = df['Vendor_number'].iloc[0]
        order=vendors[vendor]
    elif lod == 'Material Group':
        matgroup = df['Material_Group'].iloc[0]
        order=materiels[str(matgroup)]
    elif lod == 'Material':
        mat = df['Material'].iloc[0]
        order=mats[str(mat)]
    elif lod == 'Full':
        order = (1,0,0)
    else:
        order=(1,0,0)
    
    df.sort_values(by='Scheduled_relevant_delivery_date')
    df3 = df.groupby([pd.Grouper(key='Scheduled_relevant_delivery_date', freq='M')])['late_bin'].agg(['sum','count']).reset_index()
    df3['latepercentage']=df3['sum']/df3['count']
    df3['latepercentage'] = df3['latepercentage'].fillna(0)


    
    p_values = order[0]
    d_values = order[1]
    q_values = order[2]
    warnings.filterwarnings("ignore")
    
    
    series = df3['latepercentage'].squeeze()
    a = series.values
    b=a.astype('float32')
    print(series)
    results = evaluate_arima_model(b, order)
    optimal_order = order
    mse=results['mse']
    rmse=results['rmse']
    test=results['test']
    fulldf=results['df_full']
    predictions=results['predictions']
    residuals = [test[i]-predictions[i] for i in range(len(predictions))]
    
    mean = sum(residuals) / len(residuals)
    variance = sum([((x - mean) ** 2) for x in residuals]) / len(residuals)
    std = variance ** 0.5
    
    
    #residuals = DataFrame(residuals)
    
    
    model = ARIMA(df3['latepercentage'], order=optimal_order)
    model_fit = model.fit()
    #predictions=model_fit.predict(start=((df3.shape[0])),end=((df3.shape[0])+monthstopredict-1))
    
    startingpoint = df3['Scheduled_relevant_delivery_date'].iat[-1]
    months = []
    for x in range(monthstopredict):
        y=x+1
        months.append(startingpoint + relativedelta(months=+y))
        
        
    predictions=model_fit.predict(start=((df3.shape[0])),end=((df3.shape[0])+monthstopredict-1))
    finalpred=[]
    
   
    
    for x in predictions:
        if x < 0:
            finalpred.append(0)
        elif x > 1:
            finalpred.append(1)
        else:
            finalpred.append(x)
    
    upperbound=[]
    lowerbound=[]
    
    for x in finalpred:
        upperbound.append(x+1.96*std)
        lowerbound.append(x-1.96*std)
    
    finalupperbound=[]
    finallowerbound=[]
    for x in upperbound:
        if x < 0:
            finalupperbound.append(0)
        elif x > 1:
            finalupperbound.append(1)
        else:
            finalupperbound.append(x)
            
    for x in lowerbound:
        if x < 0:
            finallowerbound.append(0)
        elif x > 1:
            finallowerbound.append(1)
        else:
            finallowerbound.append(x)
            
    dfinalpred = pd.DataFrame()
    dfinalpred['predicted_percentage_late']=finalpred
    dfinalpred['time']=months
    
    dfinalpredupper = pd.DataFrame()
    dfinalpredupper['predicted_percentage_late']=finalupperbound
    dfinalpredupper['time']=months
    
    dfinalpredlower = pd.DataFrame()
    dfinalpredlower['predicted_percentage_late']=finallowerbound
    dfinalpredlower['time']=months

    return(dfinalpred,dfinalpredupper,dfinalpredlower)

# In[16]:


def forecast_supplier_pctlate(df, vendornumber, monthstopredict):
    df['Scheduled_relevant_delivery_date'] = pd.to_datetime(df['Scheduled_relevant_delivery_date'],
                                                            format='%Y-%m-%d %H:%M:%S')
    df.sort_values(by='Scheduled_relevant_delivery_date')
    df2 = df[df['Vendor_number'] == vendornumber]
    # return(df2)
    df3 = df2.groupby([pd.Grouper(key='Scheduled_relevant_delivery_date', freq='MS')])['late_bin'].agg(
        ['sum', 'count']).reset_index()
    return (forecasting(df3, monthstopredict))


# print(forecast_supplier_pctlate(df,505993,6))
# print(forecast_supplier_pctlate(df,529485,6))
# print(forecast_supplier_pctlate(df,501967,6))
# print(forecast_supplier_pctlate(df,522865,6))


# In[ ]:


def forecast_materielgroup_pctlate(df, materielgroup, monthstopredict):
    df['Scheduled_relevant_delivery_date'] = pd.to_datetime(df['Scheduled_relevant_delivery_date'],
                                                            format='%Y-%m-%d %H:%M:%S')
    df.sort_values(by='Scheduled_relevant_delivery_date')
    df2 = df[df['Material_Group'] == materielgroup]
    df3 = df2.groupby([pd.Grouper(key='Scheduled_relevant_delivery_date', freq='MS')])['late_bin'].agg(
        ['sum', 'count']).reset_index()
    return (forecasting(df3, monthstopredict))


# print(forecast_materielgroup_pctlate(df,'73',3))


# In[ ]:


def forecast_supplier_by_materielgroup__pctlate(df, vendornumber, materielgroup, monthstopredict):
    df['Scheduled_relevant_delivery_date'] = pd.to_datetime(df['Scheduled_relevant_delivery_date'],
                                                            format='%Y-%m-%d %H:%M:%S')
    df.sort_values(by='Scheduled_relevant_delivery_date')
    df2 = df[df['Material_Group'] == materielgroup]
    df3 = df2[df2['Vendor_number'] == vendornumber]
    df4 = df3.groupby([pd.Grouper(key='Scheduled_relevant_delivery_date', freq='MS')])['late_bin'].agg(
        ['sum', 'count']).reset_index()
    # return(df4)
    return (forecasting(df4, monthstopredict))


# print(forecast_supplier_by_materielgroup__pctlate(df,551948,'73',3)))
# print(forecast_supplier_by_materielgroup__pctlate(df,551948,'73',3))
# print(forecast_supplier_by_materielgroup__pctlate(df,556264,'73',3))


# In[ ]:


def forecast_overview_pctlate(df, monthstopredict):
    df['Scheduled_relevant_delivery_date'] = pd.to_datetime(df['Scheduled_relevant_delivery_date'],
                                                            format='%Y-%m-%d %H:%M:%S')
    df.sort_values(by='Scheduled_relevant_delivery_date')
    df2 = df
    df3 = df2.groupby([pd.Grouper(key='Scheduled_relevant_delivery_date', freq='MS')])['late_bin'].agg(
        ['sum', 'count']).reset_index()
    return (forecasting(df3, monthstopredict))


# print(forecast_overview_pctlate(df,3))
