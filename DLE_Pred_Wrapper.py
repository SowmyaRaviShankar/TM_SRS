
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np
import imp
import pickle
import os
import json
#import category_encoders as ce
#from sklearn.preprocessing import LabelEncoder as le
#import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import time
import sklearn.metrics as metrics
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import time


# In[60]:


#fpath1='/jup/GK/CommonFunctions/'
#fpath1 = 'D:/ETTR/Flask/Flask_files/Process'

#fpath1 = 'H:/Ettx/ETTR_Flask/Process/'
#print("test standalone path is ",fpath1)
#os.chdir(fpath1)
#import test_standalone
#import modelprocess 
#import EDA 
#import others  

#imp.reload(test_standalone)
#imp.reload(modelprocess)
#imp.reload(EDA)
#imp.reload(others)

    

# In[83]:


def worker(dle_select3):
    global int_list
    global main_list
    
  
    #-----------------FOR INTERNAL TICKETS - STARTED--------------------------#
    def predict_DLE(select3):
        print("I am in Predict DLE of Pred Wrapper file ")
        df_tkt_final_test = pd.read_csv("H:/Ettx/ETTR_Flask_App/Flask_data/DLE_data/df_tkt_final_test_1.csv" , index_col=0)
        '''
        filename_df_tkt_final_test_1 = 'H:/Ettx/ETTR_Flask_App/Flask_data/DLE_data/df_tkt_final_test_1.pkl'
        with open(filename_df_tkt_final_test_1,'rb') as file12 :
            df_tkt_final_test = pickle.load(file12)
        ''' 
        print("The select3 value is ",dle_select3)
        print("the data is ",df_tkt_final_test.head(1))
        import time
        
        #select4 = "'"+select3
        #print("The value of select4 is ", select4)
        start=time.time()
        df_tkt_final_test_1 = df_tkt_final_test[df_tkt_final_test['MAINTENANCE_REQUEST_ID_1'] == dle_select3] 
        print("The data is ",df_tkt_final_test_1)
        #from catboost import CatBoostClassifier
        df_tkt_final_1_sub= df_tkt_final_test_1[df_tkt_final_test_1.columns.difference(['TTTR_DAYS','TTTR_HOURS','MAINTENANCE_REQUEST_ID_1','MAINTENANCE_REQUEST_NRTV_dgns','MAINTENANCE_REQUEST_NRTV_dgns_cleaned','TROUBLE_DESC_cleaned','TROUBLE_DESC','CUSTOMER_NAME','TIMESTAMP_crt_Is_month_start',      
'TIMESTAMP_crt_Is_month_end'  , 'CUST_RPTD_SERVICE_ID',        'TIMESTAMP_crt_Is_year_start',         'TIMESTAMP_crt_Is_year_end','TIMESTAMP_crt_year','weekend','CUST_BONDING_INDICATOR','POWER_TO_CPE','SERVICE_CLASSIFICATION','SYS_DETM_SVC_CLASSIFICATION','MR_INITIATOR_DISPATCH_AUTH','ALARM_OVERALL_STATE','CUST_PREF_NOTIFY',
'SWITCH_TYPE','INTTEST_AUTH_TYPE','INTTEST_AUTH_TYPE_AT_CREATE','MR_TYPE_CODE','CONCLUSION_CATEGORY_dgns','TECH_PRIORITY','MCO_CLLI',
'NCO_CLLI','FR_SVC_INDICATOR','IGEMS_IND','BRIDGE_SYS_TKT_ID','ALARM_AT_CREATE_INDICATOR','CUST_PREF_NOTF','MR_CREATE_DATE_1',
'MR_CLEAR_DATE_1'])]

        X = df_tkt_final_1_sub.drop('TTTR_CAT',axis = 1)
        y = df_tkt_final_1_sub['TTTR_CAT']
        
        filename_Gradient_Boost_Model_Internal = 'H:/Ettx/ETTR_Flask_App/Flask_data/DLE_data/model_Gradient_BoostInternal.pkl'
        with open(filename_Gradient_Boost_Model_Internal,'rb') as file13 :
            gbm = pickle.load(file13)
            
            
        test_X = X
        test_y = y

        from sklearn import metrics
        #pred_test = pd.DataFrame(gbm.predict(test_X), index=test_X.index)
        
        # Get Accuracy
        gb_test_pred = pd.DataFrame( { 'actual':  test_y,'predicted': gbm.predict( test_X ) } )
        pred_test = gb_test_pred['predicted']
        print("Test Accuracy:")
        accuracy = metrics.accuracy_score( gb_test_pred.actual, gb_test_pred.predicted )
        
        # Get Probabilities
        class_1 = pd.DataFrame(gbm.classes_)
        preds_proba = pd.DataFrame(gbm.predict_proba(test_X))
        preds_proba = preds_proba.T
        preds_proba.shape , class_1.shape
        prob_pred = pd.concat([class_1, preds_proba], axis=1)
        prob_pred.columns = ['class', 'probability']
        prob_pred1 = prob_pred.sort_values('probability',ascending=False)
        prob_pred1['probability'] = pd.Series(["{0:.2f}%".format(val * 100) for val in prob_pred1['probability']], index = prob_pred1.index)
        prob_pred2 = prob_pred1.iloc[0:3,]
        prob_pred2 = prob_pred2.values.tolist()
        
        ###Display
        CTB1_test_pred = pd.concat([test_y, pred_test], axis=1)
        CTB1_test_pred.columns = ['actual', 'predicted']
        actual = CTB1_test_pred.actual.iloc[0]
        print(actual)
        end = time.time()
        print(f"Runtime of catboost is {end - start}")
        return pred_test, accuracy, prob_pred2, actual
    
        
        
      
    
 ###################################################   
   
    print("I am in Pred Wrapper")
    with open("H:\\Ettx\\ETTR_Flask_App\\Flask_data\\DLE_data\\config_VariableList.json") as json_file:
        data = json.load(json_file)
    #print("I am in ticket details, domain is -  ",select1)
    dle_select1 = "Main"
    
    if(dle_select1 == "Main"):
        print("I am in ticket details, domain Main")
        int_list=[]
        ######Get WebEx related variables######   
        for v in data['VariableList']['Main']:
            int_list.append(v)
        print("I am in index- list of Main variables  ",int_list)
        
        #adiavpn_data = pd.read_csv("H:\\Ettx\\ETTR_Flask\\Flask_data\\df_tkt_ref_2020_Sub_2_Internal.csv",encoding = 'latin1')
        adiavpn_data = pd.read_csv("H:\\Ettx\\ETTR_Flask_App\\Flask_data\\DLE_data\\df_dgns_test_final.csv")
        print(int_list)
        #select5 = select3[1:]
        df_ticket = adiavpn_data[adiavpn_data["MAINTENANCE_REQUEST_ID_1"] == dle_select3]
        
        #df_ticket = adiavpn_data[adiavpn_data[int_list[0]] == select3]
        #mydata = df_ticket.values.tolist()
        #df1 = df_ticket
        
    
    df1 = df_ticket
    print("Final test ticket data selected")
    print(df1.shape)
    print(df1)
    #print(df1[df1['TICKET__#']])
    
    ###########Preprocess - Prediction###########
    #if(select1 == "Main"):
        #df1 = df1.drop(columns='Unnamed: 0')
        #df1 = preprocess_Main(df1)
    pred, accuracy, prob_pred2, actual = predict_DLE(dle_select3)
    return pred, accuracy, prob_pred2, actual


# In[84]:


if __name__=='__main__':
    res1, res2, res3, res4 = worker(dle_select3)

