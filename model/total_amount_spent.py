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
import xgboost as xgb
import lightgbm as lgb

class model(object):
    def __init__(self, train_data, train_label, test1_data, test2_data, folds, seed):
        self.train_data = train_data.fillna(0)
        self.train_label = train_label
        self.test1_data = test1_data.fillna(0)
        self.test2_data = test2_data.fillna(0)
        self.features_ = train_data.drop(columns=['acc_id', 'week']).columns
        self.true_index = defaultdict()
        self.folds = folds
        self.seed = seed
        self.model = None
    
    
    
    def labeling_tas(self, payment):
        self.train_data = pd.merge(self.train_data, self.train_label, how='left', on='acc_id').dropna().reset_index(drop=True)
        self.train_data['adjust_survival_time'] = self.train_data['survival_time'] + self.train_data['week']*7
        
        def payment_transform(data):
            data['week'] = (data['day']-1)//7 + 1
            data = data.groupby(['acc_id', 'week']).sum().reset_index()
            return data
        payment = payment_transform(payment)
        
        def temp_func(data):
            if data['adjust_survival_time']>64:
                over_value = data['adjust_survival_time'] - 64
                data['adjust_survival_time'] = data['survival_time'] - over_value
            else:
                data['adjust_survival_time'] = data['survival_time']
            return data['adjust_survival_time']
        
        self.train_data['adjust_survival_time'] = self.train_data[['survival_time', 'adjust_survival_time']].apply(temp_func, axis=1)
        self.train_data['payment'] = 0
        
        for week in range(1, 4):
            self.train_data.loc[self.train_data['week']==week, 'payment'] = pd.merge(self.train_data.loc[self.train_data['week']==week, 'acc_id'], 
                                                                                     payment[payment['week']>week].groupby(
                                                                                         ['acc_id'])['amount_spent'].sum().reset_index().rename(columns={'amount_spent':'payment'}), 
                                                                                     how='left', on='acc_id')['payment']
            self.train_data = self.train_data.fillna(0)
        else:
            self.train_data['total_amount_spent'] = self.train_data['amount_spent'] * self.train_data['adjust_survival_time'] + self.train_data['payment']
            self.train_data = self.train_data.drop(columns=['amount_spent', 'payment'])
            for week in range(1, 5):
                self.train_data.loc[self.train_data['week']==week, 'survival_time'] = np.minimum(64, self.train_data.loc[self.train_data['week']==week, 'survival_time'] + 7*(4-week))
            else:
                self.train_data = self.train_data.drop(columns=['adjust_survival_time', 'survival_time'])
                
            
    def train_fs(self, params, iteration):
        np.random.seed(self.seed)
        LABEL='total_amount_spent'
        kf = KFold(n_splits=self.folds, random_state=self.seed, shuffle=True)
        
        for idx, (trn_idx, val_idx) in enumerate(kf.split(self.train_data)):
            
            temp_list = list()
            round_basis=1;SIZE=46
            for round_value in np.round(self.train_data['total_amount_spent'], round_basis).value_counts().index:
                temp_df_index = self.train_data.loc[trn_idx, 'total_amount_spent'].index[np.round(self.train_data.loc[trn_idx, 'total_amount_spent'], round_basis).isin([round_value])]
                try:
                    temp_df_index = np.random.choice(temp_df_index, size=SIZE, replace=False)
                except:
                    pass
                temp_list.extend(temp_df_index.tolist())
            else:
                temp_train_data = self.train_data.loc[sorted(temp_list)]
                temp_train_data = temp_train_data[temp_train_data['total_amount_spent']<40].reset_index(drop=True)
                
                trn_label = temp_train_data[LABEL]
                val_label = self.train_data.loc[val_idx, LABEL]
                train_df = lgb.Dataset(temp_train_data[self.features_], label=trn_label)
                valid_df = lgb.Dataset(self.train_data.loc[val_idx, self.features_], label=val_label)

                lgb_model = lgb.train(params, train_df, iteration, valid_sets = [train_df, valid_df], early_stopping_rounds=1000, verbose_eval=3000)
                feature_imp = pd.DataFrame(sorted(zip(lgb_model.feature_importance(), self.features_)), columns=['Value','Feature'])            
                feature_imp.to_csv('../model/feature_importance_tas.csv')
                feature_imp = feature_imp.loc[feature_imp['Value']>25, 'Feature'].tolist()
                self.features_ = feature_imp
                break

    
    
    def train_tas(self, params, iteration, model):
        self.model=model
        np.random.seed(self.seed)
        LABEL='total_amount_spent'
        kf = KFold(n_splits=self.folds, random_state=self.seed, shuffle=True)
        
        for idx, (trn_idx, val_idx) in enumerate(kf.split(self.train_data)):
            
            temp_list = list()
            round_basis=1;SIZE=46
            for round_value in np.round(self.train_data['total_amount_spent'], round_basis).value_counts().index:
                temp_df_index = self.train_data.loc[trn_idx, 'total_amount_spent'].index[np.round(self.train_data.loc[trn_idx, 'total_amount_spent'], round_basis).isin([round_value])]
                try:
                    temp_df_index = np.random.choice(temp_df_index, size=SIZE, replace=False)
                except:
                    pass
                temp_list.extend(temp_df_index.tolist())
            else:
                temp_train_data = self.train_data.loc[sorted(temp_list)]
                temp_train_data = temp_train_data[temp_train_data['total_amount_spent']<40].reset_index(drop=True)
            
                
                trn_label = temp_train_data[LABEL]
                val_label = self.train_data.loc[val_idx, LABEL]
                if self.model =='lgb':
                    train_df = lgb.Dataset(temp_train_data[self.features_], label=trn_label)
                    valid_df = lgb.Dataset(self.train_data.loc[val_idx, self.features_], label=val_label)

                    lgb_model = lgb.train(params, train_df, iteration, valid_sets = [train_df, valid_df], early_stopping_rounds=1000, verbose_eval=3000)
                    joblib.dump(lgb_model, "save_model_tas_joblib/lgb_tas_" + str(self.seed) + '_' + str(idx) + '.ckpt')
                
                elif self.model =='xgb':
                    train_df = xgb.DMatrix(temp_train_data[self.features_], label=trn_label)
                    valid_df = xgb.DMatrix(self.train_data.loc[val_idx, self.features_], label=val_label)

                    xgb_model = lgb.train(params, train_df, iteration, valid_sets = [train_df, valid_df], early_stopping_rounds=1000, verbose_eval=3000)
                    joblib.dump(xgb_model, "save_model_tas_joblib/xgb_tas_" + str(self.seed) + '_' + str(idx) + '.ckpt')
                
                    
    def model_load_infer_oof(self, model):
        self.model=model
        LABEL='total_amount_spent'
        oof = np.zeros(len(self.train_data))
        kf = KFold(n_splits=self.folds, random_state=self.seed, shuffle=True)
        
        for idx, (trn_idx, val_idx) in enumerate(kf.split(self.train_data)):
            if self.model=='lgb':
                lgb_model = joblib.load("save_model_tas_joblib/lgb_tas_" + str(self.seed) + "_" + str(idx) + ".ckpt")
                oof[val_idx] = lgb_model.predict(self.train_data.loc[val_idx, self.features_])
            elif self.model=='xgb':
                xgb_model = joblib.load("save_model_tas_joblib/xgb_tas_" + str(self.seed) + "_" + str(idx) + ".ckpt")
                oof[val_idx] = xgb_model.predict(xgb.DMatrix(self.train_data.loc[val_idx, self.features_]))
        else:
            oof = pd.concat([self.train_data, pd.DataFrame(oof, columns=['infer_total_amount_spent'])], 1)
            oof = oof.loc[oof['week']==4, ['acc_id', 'total_amount_spent', 'infer_total_amount_spent']].reset_index(drop=True)
            self.oof_tas = oof
            return self.oof_tas
    
    def model_load_infer_pred(self):
        test1 = self.test1_data.loc[self.test1_data['week']==4].reset_index(drop=True)
        test2 = self.test2_data.loc[self.test2_data['week']==4].reset_index(drop=True)
        pred1 = np.zeros([len(test1), self.folds])
        pred2 = np.zeros([len(test2), self.folds])
        
        for idx in range(self.folds):
            if self.model=='lgb':
                lgb_model = joblib.load("save_model_tas_joblib/lgb_tas_" + str(self.seed) + "_" + str(idx) + ".ckpt")
                pred1[:, idx] = lgb_model.predict(test1[self.features_])
                pred2[:, idx] = lgb_model.predict(test2[self.features_])
            elif self.model=='xgb':
                xgb_model = joblib.load("save_model_tas_joblib/xgb_tas_" + str(self.seed) + "_" + str(idx) + ".ckpt")
                pred1[:, idx] = xgb_model.predict(xgb.DMatrix(test1[self.features_]))
                pred2[:, idx] = xgb_model.predict(xgb.DMatrix(test2[self.features_]))
        else:
            pred1 = pd.concat([test1['acc_id'], pd.DataFrame(pred1)], 1)
            pred2 = pd.concat([test2['acc_id'], pd.DataFrame(pred2)], 1)
            return pred1, pred2

    
    def load(self, return_data):
        if return_data=='train':
            return self.train_data
        elif return_data=='test1':
            return self.test1_data
        elif return_data=='test2':
            return self.test2_data
        elif return_data=='model_st':
            return self.lgb_model_st
        elif return_data=='model_tas':
            return self.lgb_model_tas
        elif return_data=='true_dict':
            return self.true_index
        elif return_data=='feature':
            return self.features_
    


if __name__ == '__main__':        
    FOLDS=5
    SEED=42
    PARAMS_TAS = {
        'objective':'regression',
        "boosting": "gbdt",
        'learning_rate': 0.02,
        'subsample' : 0.6,
        'sumsample_freq':1,
        'colsample_bytree':0.221856,
        'max_depth': 8,
        'max_bin':255,
        "lambda_l1": 0.25,
        "lambda_l2": 1,
        'min_child_weight': 0.2,
        'min_child_samples': 20,
        'min_gain_to_split':0.02,
        'min_data_in_bin':3,
        'bin_construct_sample_cnt':5000,
        'cat_l2':10,
        'verbose':-1,
        'nthread':-1,
        'metrics':'rmse',
        'seed':SEED
    }

    train = pd.read_csv('../preprocess/train.csv')
    test1 = pd.read_csv('../preprocess/test1.csv')
    test2 = pd.read_csv('../preprocess/test2.csv')

    train_payment = pd.read_csv('../raw/train_payment.csv')
    train_label = pd.read_csv('../raw/train_label.csv')


    main_model = model(train, train_label, test1, test2, FOLDS, SEED)
    print('start main_model')

    # total amount spent
    main_model.labeling_tas(train_payment)
    main_model.train_fs(PARAMS_TAS, 50000)
    
    # lgb
    select_model='lgb'
    main_model.train_tas(PARAMS_TAS, 50000, select_model)

    oof_tas = main_model.model_load_infer_oof(select_model)
    oof_tas.to_csv('../predict/oof_tas_' + select_model + '.csv', index=False)
    pred_tas1, pred_tas2 = main_model.model_load_infer_pred()
    pred_tas1.to_csv('../predict/pred_tas1_' + select_model + '.csv', index=False)
    pred_tas2.to_csv('../predict/pred_tas2_' + select_model + '.csv', index=False)

    pred_st1 = pd.read_csv('../predict/pred_st1_lgb.csv')
    pred_st2 = pd.read_csv('../predict/pred_st2_lgb.csv')

    def pred_transform(st, tas, acc_id):
        st_t = st.copy()
        tas_t = tas.copy()
        
        st_t['survival_time'] = st_t.drop(columns='acc_id').median(1)
        tas_t['amount_spent'] = tas_t.drop(columns='acc_id').mean(1)
        
        pred_df = pd.merge(st_t, tas_t, how='left', on='acc_id')[['acc_id', 'survival_time', 'amount_spent']]
        pred_df['amount_spent'] = pred_df['amount_spent']/pred_df['survival_time']
        pred_df.loc[pred_df['amount_spent']<0, 'amount_spent'] = 0
        
        pred_df = pred_df[pred_df['acc_id'].isin(acc_id)]
        return pred_df

    pred_transform(pred_st1, pred_tas1, test1['acc_id'].unique()).to_csv('../predict/test1_predict.csv', index=False)
    pred_transform(pred_st2, pred_tas2, test2['acc_id'].unique()).to_csv('../predict/test2_predict.csv', index=False)
    