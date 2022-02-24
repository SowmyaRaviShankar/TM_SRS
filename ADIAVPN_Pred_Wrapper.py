
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


def worker(adiavpn_select3,adiavpn_select1):
    global int_list
    global main_list
    
    
        
    def preprocess_Internal(df_2020tkt):
        #
        
        
        #
        
        
        df_tkt_ref_2020_Sub_2_Internal = df_2020tkt.loc[:, ~df_2020tkt.columns.str.contains('^Unnamed')]
        import pickle
        filename_1911_1913_CatFieldSelection = 'H:/Ettx/ETTR_Flask_App/Encoders/1911_1913_'+'CatFieldSelection.pkl'
        with open(filename_1911_1913_CatFieldSelection,'rb') as file :
            cat_cols_1 = pickle.load(file)
        df_category_20 = df_tkt_ref_2020_Sub_2_Internal[cat_cols_1]
        df_category_20.columns
        df_category_20 = pd.DataFrame(data = df_category_20)
        from sklearn.preprocessing import LabelEncoder
        import seaborn as sns
        ## Import Pickle file of Label Encoder for Testing Data
        filename_Int_le = 'H:/Ettx/ETTR_Flask_App/Encoders/le_Training.pkl'
        with open(filename_Int_le,'rb') as file :
            le = pickle.load(file)
        #le_20 = LabelEncoder()
        df1_cat_LE_20 = df_category_20.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')
        #Convert Categorical data to Label Encoder
        df1_cat_LE_20[cat_cols_1] = df1_cat_LE_20[cat_cols_1].astype(str)
        # Combine all columns
        df_tkt_final_20 = pd.DataFrame()
        #Ticket Number column
        df_tkt_final_20['TICKET_#'] = df_tkt_ref_2020_Sub_2_Internal['TICKET__#']
        #Other columns to join
        numndate_20 = df_tkt_ref_2020_Sub_2_Internal[['INBOUND_CALL_COUNT', 'EVENTMESSAGECOUNT', 'TICKET_OPENEDYear', 'TICKET_OPENEDMonth', 'TICKET_OPENEDWeek', 'TICKET_OPENEDDay', 'TICKET_OPENEDDayofweek', 'TICKET_OPENEDDayofyear', 'TICKET_OPENEDElapsed','TICKET_OPENED','TICKET_OPENEDIs_month_end', 'TICKET_OPENEDIs_month_start', 'TICKET_OPENEDIs_quarter_end', 'TICKET_OPENEDIs_quarter_start', 'TICKET_OPENEDIs_year_end', 'TICKET_OPENEDIs_year_start','TIME_ELAPSED_FOR_MONITORING_left', 'TENTATIVE_TTTR', 'NEXT_CHECK', 'LAST_CHECK']]
        df_tkt_final_20 = df_tkt_final_20.join(numndate_20)
        #Add TARGET column- tttr_cat
        target_col_20 = df_tkt_ref_2020_Sub_2_Internal['TTTR_CAT']
        #Add LABEL ENCODED Categorical columns
        df_tkt_final_20 = df_tkt_final_20.join(df1_cat_LE_20)
        #Final dataframe created
        df_tkt_final_20 = df_tkt_final_20.join(target_col_20)
        print(df_tkt_final_20.shape)
        df_tkt_final_20.columns
        df_tkt_final_20['TICKET_OPENEDMonth'] = df_tkt_final_20['TICKET_OPENEDMonth'].astype(str)
        df_tkt_final_20['TICKET_OPENEDWeek'] = df_tkt_final_20['TICKET_OPENEDWeek'].astype(str)
        df_tkt_final_20['TICKET_OPENEDDayofweek'] = df_tkt_final_20['TICKET_OPENEDDayofweek'].astype(str)

        return df_tkt_final_20

    
    #-----------------FOR INTERNAL TICKETS - STARTED--------------------------#
    def predict_Internal(df_tkt_final_20):
        df_tkt_2020_final = df_tkt_final_20.copy()
        filename_Model_Build_Features = 'H:/Ettx/ETTR_Flask_App/Encoders/Model_Building_'+'Features.pkl'
        with open(filename_Model_Build_Features,'rb') as file :
            selected_feature  = pickle.load(file)


        filename_Int_cat ='H:/Ettx/ETTR_Flask_App/Encoders/model_'+'CatboostInternal.pkl'
        with open(filename_Int_cat,'rb') as file :
            model = pickle.load(file)

        import time
        start=time.time()
    
        from catboost import CatBoostClassifier

        test_X = df_tkt_2020_final[selected_feature] 
        test_y = df_tkt_2020_final['TTTR_CAT']

        pred_test = pd.DataFrame(model.predict(test_X), index=test_X.index)
        
        class_1 = pd.DataFrame(model.classes_)
        preds_proba = pd.DataFrame(model.predict_proba(test_X))
        preds_proba = preds_proba.T
        preds_proba.shape , class_1.shape
        prob_pred = pd.concat([class_1, preds_proba], axis=1)
        prob_pred.columns = ['class', 'probability']
        prob_pred1 = prob_pred.sort_values('probability',ascending=False)
        prob_pred1['probability'] = pd.Series(["{0:.2f}%".format(val * 100) for val in prob_pred1['probability']], index = prob_pred1.index)
        prob_pred2 = prob_pred1.iloc[0:3,]
        prob_pred2 = prob_pred2.values.tolist()
        
        CTB1_test_pred = pd.concat([test_y, pred_test], axis=1)
    
        CTB1_test_pred = pd.concat([test_y, pred_test], axis=1)
        CTB1_test_pred.columns = ['actual', 'predicted']
        actual = CTB1_test_pred.actual.iloc[0]


        print("Test Accuracy:")
        accuracy = metrics.accuracy_score(  CTB1_test_pred.actual, CTB1_test_pred.predicted )
        print(accuracy)
        end=time.time()
        print(f"Runtime of catboost is {end - start}")
        return pred_test, accuracy, prob_pred2, actual

    
    #-----------------FOR MAIN TICKETS - STARTED--------------------------#
   
    def preprocess_Main(df_2020tkt):
        df_tkt_ref_2020_Sub_2_Internal = df_2020tkt.loc[:, ~df_2020tkt.columns.str.contains('^Unnamed')]
        import pickle
        filename_1911_1913_CatFieldSelection = 'H:/Ettx/ETTR_Flask_App/Encoders/1911_1913_'+'CatFieldSelection.pkl'
        with open(filename_1911_1913_CatFieldSelection,'rb') as file :
            cat_cols_1 = pickle.load(file)
        df_category_20 = df_tkt_ref_2020_Sub_2_Internal[cat_cols_1]
        df_category_20.columns
        df_category_20 = pd.DataFrame(data = df_category_20)
        from sklearn.preprocessing import LabelEncoder
        import seaborn as sns
        ## Import Pickle file of Label Encoder for Testing Data
        filename_Int_le = 'H:/Ettx/ETTR_Flask_App/Encoders/le_Standaloneit2.pkl'
        with open(filename_Int_le,'rb') as file :
            le = pickle.load(file)
        #le_20 = LabelEncoder()
        df1_cat_LE_20 = df_category_20.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')
        #Convert Categorical data to Label Encoder
        df1_cat_LE_20[cat_cols_1] = df1_cat_LE_20[cat_cols_1].astype(str)
        # Combine all columns
        df_tkt_final_20 = pd.DataFrame()
        #Ticket Number column
        df_tkt_final_20['TICKET_#'] = df_tkt_ref_2020_Sub_2_Internal['TICKET__#']
        #Other columns to join
        numndate_20 = df_tkt_ref_2020_Sub_2_Internal[['INBOUND_CALL_COUNT', 'EVENTMESSAGECOUNT', 'TICKET_OPENEDYear', 'TICKET_OPENEDMonth', 'TICKET_OPENEDWeek', 'TICKET_OPENEDDay', 'TICKET_OPENEDDayofweek', 'TICKET_OPENEDDayofyear', 'TICKET_OPENEDElapsed','TICKET_OPENED','TICKET_OPENEDIs_month_end', 'TICKET_OPENEDIs_month_start', 'TICKET_OPENEDIs_quarter_end', 'TICKET_OPENEDIs_quarter_start', 'TICKET_OPENEDIs_year_end', 'TICKET_OPENEDIs_year_start','TIME_ELAPSED_FOR_MONITORING_left', 'TENTATIVE_TTTR', 'NEXT_CHECK', 'LAST_CHECK']]
        df_tkt_final_20 = df_tkt_final_20.join(numndate_20)
        #Add TARGET column- tttr_cat
        target_col_20 = df_tkt_ref_2020_Sub_2_Internal['TTTR_CAT']
        #Add LABEL ENCODED Categorical columns
        df_tkt_final_20 = df_tkt_final_20.join(df1_cat_LE_20)
        #Final dataframe created
        df_tkt_final_20 = df_tkt_final_20.join(target_col_20)
        print(df_tkt_final_20.shape)
        df_tkt_final_20.columns
        df_tkt_final_20['TICKET_OPENEDMonth'] = df_tkt_final_20['TICKET_OPENEDMonth'].astype(str)
        df_tkt_final_20['TICKET_OPENEDWeek'] = df_tkt_final_20['TICKET_OPENEDWeek'].astype(str)
        df_tkt_final_20['TICKET_OPENEDDayofweek'] = df_tkt_final_20['TICKET_OPENEDDayofweek'].astype(str)

        return df_tkt_final_20

    
    #-----------------FOR INTERNAL TICKETS - STARTED--------------------------#
    def predict_Main(df_tkt_final_20):
        df_tkt_2020_final = df_tkt_final_20.copy()
        filename_Model_Build_Features = 'H:/Ettx/ETTR_Flask_App/Encoders/Model_Building_'+'Features.pkl'
        with open(filename_Model_Build_Features,'rb') as file :
            selected_feature  = pickle.load(file)


        filename_Int_cat ='H:/Ettx/ETTR_Flask_App/Encoders/model_Catbooststandalone.pkl'
        with open(filename_Int_cat,'rb') as file :
            model = pickle.load(file)

        import time
        start=time.time()
    
        from catboost import CatBoostClassifier

        test_X = df_tkt_2020_final[selected_feature] 
        test_y = df_tkt_2020_final['TTTR_CAT']

        pred_test = pd.DataFrame(model.predict(test_X), index=test_X.index)
        
        class_1 = pd.DataFrame(model.classes_)
        preds_proba = pd.DataFrame(model.predict_proba(test_X))
        preds_proba = preds_proba.T
        preds_proba.shape , class_1.shape
        prob_pred = pd.concat([class_1, preds_proba], axis=1)
        prob_pred.columns = ['class', 'probability']
        prob_pred1 = prob_pred.sort_values('probability',ascending=False)
        prob_pred1['probability'] = pd.Series(["{0:.2f}%".format(val * 100) for val in prob_pred1['probability']], index = prob_pred1.index)
        prob_pred2 = prob_pred1.iloc[0:3,]
        prob_pred2 = prob_pred2.values.tolist()
        
        CTB1_test_pred = pd.concat([test_y, pred_test], axis=1)
    
        CTB1_test_pred = pd.concat([test_y, pred_test], axis=1)
        CTB1_test_pred.columns = ['actual', 'predicted']
        actual = CTB1_test_pred.actual.iloc[0]


        print("Test Accuracy:")
        accuracy = metrics.accuracy_score(  CTB1_test_pred.actual, CTB1_test_pred.predicted )
        print(accuracy)
        end=time.time()
        print(f"Runtime of catboost is {end - start}")
        return pred_test, accuracy, prob_pred2, actual
    #-----------------FOR MAIN TICKETS - ENDED--------------------------#
    
    
   

    with open("H:\\Ettx\\ETTR_Flask_App\\Flask_data\\ADIAVPN_data\\config_VariableList.json") as json_file:
        data = json.load(json_file)
    #print("I am in ticket details, domain is -  ",select1)
    if(adiavpn_select1 == "Internal"):
        print("I am in ticket details, domain Internal")
        int_list=[]
        ######Get WebEx related variables######   
        for v in data['VariableList']['Internal']:
            int_list.append(v)
        print("I am in index- list of Internal variables  ",int_list)
        
        #adiavpn_data = pd.read_csv("H:\\Ettx\\ETTR_Flask\\Flask_data\\df_tkt_ref_2020_Sub_2_Internal.csv",encoding = 'latin1')
        adiavpn_data = pd.read_csv("H:\\Ettx\\ETTR_Flask_App\\Flask_data\\ADIAVPN_data\\df_tkt_ref_2020_Sub_2_Internal_New.csv",encoding = 'latin1')
        print(int_list)
        df_ticket = adiavpn_data[adiavpn_data[int_list[0]] == adiavpn_select3]
        #mydata = df_ticket.values.tolist()
        #df1 = df_ticket
        
    else:
        #adiavpn_data = pd.read_csv("H:\\Ettx\\ETTR_Flask\\Flask_data\\df_tkt_ref_2020_Sub_2_Standalone.csv",encoding = 'latin1')
        adiavpn_data = pd.read_csv("H:\\Ettx\\ETTR_Flask_App\\Flask_data\\ADIAVPN_data\\df_tkt_ref_2020_Sub_2_Standalone_New.csv",encoding = 'latin1')
        main_list=[]
        for v in data['VariableList']['Main']:
            main_list.append(v)
        print("I am in index- list of Main variables  ",main_list)
        
        print(main_list)
        df_ticket = adiavpn_data[adiavpn_data[main_list[0]] == adiavpn_select3]
        #mydata = df_ticket.values.tolist()
        #df1 = df_ticket
    
    df1 = df_ticket
    print("Final test ticket data selected")
    print(df1.shape)
    print(df1)
    #print(df1[df1['TICKET__#']])
    
    ###########Preprocess - Prediction###########
    if(adiavpn_select1 == "Main"):
        df1 = df1.drop(columns='Unnamed: 0')
        df1 = preprocess_Main(df1)
        pred, accuracy, prob_pred2, actual = predict_Main(df1)
    else:
        df1 = df1.drop(columns='Unnamed: 0')
        df1 = preprocess_Internal(df1)
        pred, accuracy, prob_pred2, actual = predict_Internal(df1)
        
    return pred, accuracy, prob_pred2, actual


# In[84]:


if __name__=='__main__':
    res1, res2, res3, res4 = worker(adiavpn_select3)

