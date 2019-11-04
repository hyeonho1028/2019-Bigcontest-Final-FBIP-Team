import os
import gc
from collections import defaultdict

import pandas as pd
import numpy as np


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import gc
import copy
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import joblib
import lightgbm as lgb
import tqdm



SEED=42

class data_transform(object):
    def __init__(self, data):
        self.data = data

    def create_week(self):
        self.data['week'] = (self.data['day']-1)//7 + 1
        return self.data
    
    def activity_transform(self):
        temp_df = self.data
        groupby_dict = defaultdict()
        
        temp_df = pd.get_dummies(temp_df)
        
        for feature in temp_df.columns:
            if feature == 'acc_id' or feature == 'week':
                pass
            elif feature == 'day' or feature == 'char_id':
                groupby_dict[feature] = 'nunique'
            else:
                groupby_dict[feature] = ['sum', 'mean', 'min', 'max']
        else:        
            temp_df = temp_df.groupby(['acc_id', 'week']).agg(groupby_dict).reset_index()
            temp_df.columns = [i+j for i,j in temp_df.columns.ravel()]
        return temp_df
    
    def payment_transform(self):
        output_df = self.data
        groupby_dict = defaultdict()
        
        for feature in output_df.columns:
            if feature == 'acc_id' or feature == 'week':
                pass
            elif feature == 'day':
                groupby_dict[feature] = 'nunique'
            else:
                groupby_dict[feature] = ['sum', 'mean', 'min', 'max']
        else:
            output_df = output_df.groupby(['acc_id', 'week']).agg(groupby_dict).reset_index()
            output_df.columns = [i+j for i,j in output_df.columns.ravel()]
        
        return output_df
    
    def trade_transform(self):
        output_df = self.data
        groupby_dict = defaultdict()
        groupby_dict2 = defaultdict()
        
        output_df['time'] = output_df['time'].apply(lambda x: str(x)[:2])
        output_df[['time', 'type', 'server']] = output_df[['time', 'type', 'server']].astype(object)
        output_df = pd.get_dummies(output_df)
        output_df2 = output_df.copy()
        
        output_df = output_df.rename(columns={'source_acc_id':'acc_id'})
        output_df2 = output_df2.rename(columns={'target_acc_id':'acc_id'})
        
        for feature in output_df.columns:
            if feature == 'acc_id' or feature == 'week':
                pass
            elif feature in ['day', 'item_type', 'source_char_id', 'target_char_id', 'target_acc_id']:
                groupby_dict[feature] = 'nunique'
            else:
                groupby_dict[feature] = ['sum', 'mean', 'min', 'max']
        else:
            output_df = output_df.groupby(['acc_id', 'week']).agg(groupby_dict).reset_index()
            output_df.columns = [i+j for i,j in output_df.columns.ravel()]
        
        for feature in output_df2.columns:
            if feature == 'acc_id' or feature == 'week':
                pass
            elif feature in ['day', 'item_type', 'source_char_id', 'target_char_id', 'source_acc_id']:
                groupby_dict2[feature] = 'nunique'
            else:
                groupby_dict2[feature] = ['sum', 'mean', 'min', 'max']
        else:
            output_df2 = output_df2.groupby(['acc_id', 'week']).agg(groupby_dict2).reset_index()
            output_df2.columns = [i+j for i,j in output_df2.columns.ravel()]
        
        output_df = pd.merge(output_df, output_df2, how='outer', on=['acc_id', 'week'])
        
        return output_df
    
    def combat_transform(self):
        output_df = self.data
        groupby_dict = defaultdict()
        
        output_df[['server', 'class', 'level']] = output_df[['server', 'class', 'level']].astype(str)
        output_df = pd.get_dummies(output_df)
        
        for feature in output_df.columns:
            if feature == 'acc_id' or feature == 'week':
                pass
            elif feature == 'day':
                groupby_dict[feature] = 'nunique'
            elif feature == 'char_id':
                groupby_dict[feature] = ['nunique', 'size']
            else:
                groupby_dict[feature] = ['sum', 'mean', 'min', 'max']
        else:
            output_df = output_df.groupby(['acc_id', 'week']).agg(groupby_dict).reset_index()
            output_df.columns = [i+j for i,j in output_df.columns.ravel()]
        
        return output_df
    
    def pledge_transform(self):
        output_df = self.data
        groupby_dict = defaultdict()
        
        output_df[['server']] = output_df[['server']].astype(str)
        output_df = pd.get_dummies(output_df)
        
        for feature in output_df.columns:
            if feature == 'acc_id' or feature == 'week':
                pass
            elif feature in ['day', 'pledge_id']:
                groupby_dict[feature] = 'nunique'
            elif feature == 'char_id':
                groupby_dict[feature] = ['nunique', 'size']
            else:
                groupby_dict[feature] = ['sum', 'mean', 'min', 'max']
        else:
            output_df = output_df.groupby(['acc_id', 'week']).agg(groupby_dict).reset_index()
            output_df.columns = [i+j for i,j in output_df.columns.ravel()]
        return output_df

if __name__ == '__main__':
    path = '../raw/'
    train_activity = pd.read_csv(path + 'train_activity.csv').drop(columns='fishing')
    train_combat = pd.read_csv(path + 'train_combat.csv')
    train_payment = pd.read_csv(path + 'train_payment.csv')
    train_pledge = pd.read_csv(path + 'train_pledge.csv').drop(columns=['combat_play_time', 'non_combat_play_time'])
    train_trade = pd.read_csv(path + 'train_trade.csv')

    test1_activity = pd.read_csv(path + 'test1_activity.csv').drop(columns='fishing')
    test1_combat = pd.read_csv(path + 'test1_combat.csv')
    test1_payment = pd.read_csv(path + 'test1_payment.csv')
    test1_pledge = pd.read_csv(path + 'test1_pledge.csv').drop(columns=['combat_play_time', 'non_combat_play_time'])
    test1_trade = pd.read_csv(path + 'test1_trade.csv')

    test2_activity = pd.read_csv(path + 'test2_activity.csv').drop(columns='fishing')
    test2_combat = pd.read_csv(path + 'test2_combat.csv')
    test2_payment = pd.read_csv(path + 'test2_payment.csv')
    test2_pledge = pd.read_csv(path + 'test2_pledge.csv').drop(columns=['combat_play_time', 'non_combat_play_time'])
    test2_trade = pd.read_csv(path + 'test2_trade.csv')

    # train
    train_activity['game_money_change_abs'] = np.abs(train_activity['game_money_change'])
    transform = data_transform(train_activity)
    transform.create_week()
    act_train = transform.activity_transform()
    transform = data_transform(train_payment)
    transform.create_week()
    pay_train = transform.payment_transform()
    trandform = data_transform(train_trade)
    trandform.create_week()
    tra_train = trandform.trade_transform()
    trandform = data_transform(train_combat)
    trandform.create_week()
    com_train = trandform.combat_transform()
    trandform = data_transform(train_pledge)
    trandform.create_week()
    ple_train = trandform.pledge_transform()
    gc.collect()


    # test
    test1_activity['game_money_change_abs'] = np.abs(test1_activity['game_money_change'])
    transform = data_transform(test1_activity)
    transform.create_week()
    act_test1 = transform.activity_transform()
    test2_activity['game_money_change_abs'] = np.abs(test2_activity['game_money_change'])
    transform = data_transform(test2_activity)
    transform.create_week()
    act_test2 = transform.activity_transform()
    transform = data_transform(test1_payment)
    transform.create_week()
    pay_test1 = transform.payment_transform()
    transform = data_transform(test2_payment)
    transform.create_week()
    pay_test2 = transform.payment_transform()
    trandform = data_transform(test1_trade)
    trandform.create_week()
    tra_test1 = trandform.trade_transform()
    trandform = data_transform(test2_trade)
    trandform.create_week()
    tra_test2 = trandform.trade_transform()
    trandform = data_transform(test1_combat)
    trandform.create_week()
    com_test1 = trandform.combat_transform()
    trandform = data_transform(test2_combat)
    trandform.create_week()
    com_test2 = trandform.combat_transform()
    trandform = data_transform(test1_pledge)
    trandform.create_week()
    ple_test1 = trandform.pledge_transform()
    trandform = data_transform(test2_pledge)
    trandform.create_week()
    ple_test2 = trandform.pledge_transform()
    gc.collect()


    act_train.columns = [column if column in ['acc_id', 'week'] else 'act_'+str(column) for column in act_train.columns]
    pay_train.columns = [column if column in ['acc_id', 'week'] else 'pay_'+str(column) for column in pay_train.columns]
    tra_train.columns = [column if column in ['acc_id', 'week'] else 'tra_'+str(column) for column in tra_train.columns]
    com_train.columns = [column if column in ['acc_id', 'week'] else 'com_'+str(column) for column in com_train.columns]
    ple_train.columns = [column if column in ['acc_id', 'week'] else 'ple_'+str(column) for column in ple_train.columns]

    act_test1.columns = [column if column in ['acc_id', 'week'] else 'act_'+str(column) for column in act_test1.columns]
    pay_test1.columns = [column if column in ['acc_id', 'week'] else 'pay_'+str(column) for column in pay_test1.columns]
    tra_test1.columns = [column if column in ['acc_id', 'week'] else 'tra_'+str(column) for column in tra_test1.columns]
    com_test1.columns = [column if column in ['acc_id', 'week'] else 'com_'+str(column) for column in com_test1.columns]
    ple_test1.columns = [column if column in ['acc_id', 'week'] else 'ple_'+str(column) for column in ple_test1.columns]

    act_test2.columns = [column if column in ['acc_id', 'week'] else 'act_'+str(column) for column in act_test2.columns]
    pay_test2.columns = [column if column in ['acc_id', 'week'] else 'pay_'+str(column) for column in pay_test2.columns]
    tra_test2.columns = [column if column in ['acc_id', 'week'] else 'tra_'+str(column) for column in tra_test2.columns]
    com_test2.columns = [column if column in ['acc_id', 'week'] else 'com_'+str(column) for column in com_test2.columns]
    ple_test2.columns = [column if column in ['acc_id', 'week'] else 'ple_'+str(column) for column in ple_test2.columns]

    del train_activity
    del train_combat
    del train_payment
    del train_pledge
    del train_trade
    del test1_activity
    del test1_combat
    del test1_payment
    del test1_pledge
    del test1_trade
    del test2_activity
    del test2_combat
    del test2_payment
    del test2_pledge
    del test2_trade 
    gc.collect()

    print('transform complete')
    # model
    train = pd.merge(pd.merge(pd.merge(pd.merge(act_train, pay_train, how='left', on=['acc_id', 'week']), tra_train, how='left', on=['acc_id', 'week']), com_train, how='left', on=['acc_id', 'week']), ple_train, how='left', on=['acc_id', 'week']).fillna(0)
    test1 = pd.merge(pd.merge(pd.merge(pd.merge(act_test1, pay_test1, how='left', on=['acc_id', 'week']), tra_test1, how='left', on=['acc_id', 'week']), com_test1, how='left', on=['acc_id', 'week']), ple_test1, how='left', on=['acc_id', 'week']).fillna(0)
    test2 = pd.merge(pd.merge(pd.merge(pd.merge(act_test2, pay_test2, how='left', on=['acc_id', 'week']), tra_test2, how='left', on=['acc_id', 'week']), com_test2, how='left', on=['acc_id', 'week']), ple_test2, how='left', on=['acc_id', 'week']).fillna(0)

    del act_train, tra_train, pay_train, com_train, ple_train
    del act_test1, tra_test1, pay_test1, com_test1, ple_test1
    del act_test2, tra_test2, pay_test2, com_test2, ple_test2
    gc.collect()

    common_features = list(set(train.columns) & set(test1.columns) & set(test2.columns))
    train = train[common_features]
    test1 = test1[common_features]
    test2 = test2[common_features]

    all_data = pd.concat([train, test1, test2])
    
    # play style
    sum_columns = all_data.columns[[column[-3:]=='sum' for column in all_data.columns]]
    playtimesum = all_data['act_playtimesum']
    for column in sum_columns:
        if column!='act_playtimesum':
            all_data['derive' + str(column)] = all_data[column]/playtimesum


    # week
    dummy = all_data[['acc_id', 'week']]
    dummy_df = pd.get_dummies(dummy.groupby('acc_id')['week'].first()).reset_index()
    dummy_df.columns = ['acc_id', 'enter_week1', 'enter_week2', 'enter_week3', 'enter_week4']
    all_data = pd.merge(all_data, dummy_df, how='left', on='acc_id')

    # reduce memory
    def reduce_mem_usage(df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2    
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)    
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df
        
    # PCA
    pca_columns = all_data.drop(columns=['acc_id', 'week']).columns
    temp_df = pd.DataFrame(np.unique(all_data['acc_id']), columns=['acc_id'])
    for column in tqdm.tqdm(pca_columns):
        try:
            temp_pca = all_data[['acc_id', 'week', column]].groupby(['acc_id', 'week']).sum().unstack().fillna(0)
            pca_component = PCA(n_components=1, random_state=42).fit_transform(StandardScaler().fit_transform(temp_pca))
            temp_df = pd.concat([temp_df, pd.DataFrame(pca_component, columns=['pca_'+str(column)])], 1)
        except:
            pass
    else:
        all_data = reduce_mem_usage(all_data)
        temp_df = reduce_mem_usage(temp_df)
        all_data = pd.merge(all_data, temp_df, how='left', on='acc_id')
        del temp_df
        gc.collect()

    train = all_data.iloc[:train.shape[0], :]
    test1 = all_data.iloc[train.shape[0]:train.shape[0]+test1.shape[0], :].reset_index(drop=True)
    test2 = all_data.iloc[-test2.shape[0]:, :].reset_index(drop=True)

    print('saving..')

    for column in train.columns:
        train.loc[np.isinf(train[column]), column] = 0
        test1.loc[np.isinf(test1[column]), column] = 0
        test2.loc[np.isinf(test2[column]), column] = 0


    save_path = '../preprocess/'
    train.to_csv(save_path + 'train.csv', index=False)
    test1.to_csv(save_path + 'test1.csv', index=False)
    test2.to_csv(save_path + 'test2.csv', index=False)


    

    print('complete')
