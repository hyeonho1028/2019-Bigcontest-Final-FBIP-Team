import gc
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
    
    
    
    def labeling(self):
        self.train_data = pd.merge(self.train_data, self.train_label[['acc_id', 'survival_time']], how='left', on='acc_id').dropna().reset_index(drop=True)
        self.train_data = self.train_data.dropna().reset_index(drop=True)
        for week in range(1, 5):
            self.train_data.loc[self.train_data['week']==week, 'survival_time'] = np.minimum(64, self.train_data.loc[self.train_data['week']==week, 'survival_time'] + 7*(4-week))
            
    def train_fs(self, params, iteration):
        self.model=model
        LABEL='survival_time'
        for idx, true in enumerate(np.unique(self.train_data[LABEL].apply(lambda x: x if x==1 or x==64 else x//7*7).apply(lambda x: 1 if x==0 else x))):
            self.true_index[true] = idx
        else:
            self.train_data[LABEL] = self.train_data[LABEL].apply(lambda x: x if x==1 or x==64 else x//7*7).apply(lambda x: 1 if x==0 else x).apply(lambda x: self.true_index[x])
            
        skf = StratifiedKFold(n_splits=self.folds, random_state=self.seed, shuffle=True)
        rus = RandomUnderSampler(random_state=self.seed)
        for idx, (trn_idx, val_idx) in enumerate(skf.split(self.train_data, self.train_data[LABEL])):

            temp_train_data = pd.DataFrame()
            X, y = rus.fit_resample(self.train_data.drop(columns='survival_time'), self.train_data['survival_time'])
            temp_train_data = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], 1))
            temp_train_data.columns = self.train_data.columns
            
            trn_label = temp_train_data[LABEL]
            val_label = self.train_data.loc[val_idx, LABEL]
            train_df = lgb.Dataset(temp_train_data[self.features_], label=trn_label)
            valid_df = lgb.Dataset(self.train_data.loc[val_idx, self.features_], label=val_label)

            lgb_model = lgb.train(params, train_df, iteration, valid_sets = [train_df, valid_df], early_stopping_rounds=150, verbose_eval=1000)
            feature_imp = pd.DataFrame(sorted(zip(lgb_model.feature_importance(), self.features_)), columns=['Value','Feature'])
            feature_imp.to_csv('../model/feature_importance_st.csv', index=False)
            feature_imp = feature_imp.loc[feature_imp['Value']>25, 'Feature'].tolist()
            self.features_ = feature_imp
            break

    def train_st(self, params, iteration, model):
        self.model=model
        LABEL='survival_time'
            
        skf = StratifiedKFold(n_splits=self.folds, random_state=self.seed, shuffle=True)
        rus = RandomUnderSampler(random_state=self.seed)
        for idx, (trn_idx, val_idx) in enumerate(skf.split(self.train_data, self.train_data[LABEL])):

            temp_train_data = pd.DataFrame()
            X, y = rus.fit_resample(self.train_data.drop(columns='survival_time'), self.train_data['survival_time'])
            temp_train_data = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], 1))
            temp_train_data.columns = self.train_data.columns
            
            trn_label = temp_train_data[LABEL]
            val_label = self.train_data.loc[val_idx, LABEL]
            if self.model=='lgb':
                train_df = lgb.Dataset(temp_train_data[self.features_], label=trn_label)
                valid_df = lgb.Dataset(self.train_data.loc[val_idx, self.features_], label=val_label)

                lgb_model = lgb.train(params, train_df, iteration, valid_sets = [train_df, valid_df], early_stopping_rounds=150, verbose_eval=1000)
                joblib.dump(lgb_model, "save_model_st_joblib/lgb_st_" + str(self.seed) + '_' + str(idx) + '.ckpt')
                
            elif self.model=='rf':
                rf_model = RandomForestClassifier(n_estimators=1000, random_state=self.seed, max_depth=8).fit(temp_train_data[self.features_], trn_label)
                joblib.dump(rf_model, "save_model_st_joblib/rf_st_" + str(self.seed) + '_' + str(idx) + '.ckpt')
            elif self.model=='xgb':
                train_df = xgb.DMatrix(temp_train_data[self.features_], label=trn_label)
                valid_df = xgb.DMatrix(self.train_data.loc[val_idx, self.features_], label=val_label)
                
                params = {
                    'objective': 'multi:softprob',
                    'num_class': 11,
                    'n_estimators':1000,
                    'max_depth':8,
                    'learning_rate':0.1,
                    'subsample':0.9,
                    'colsample_bytree':0.9,
                    'reg_alpha':0.1,
                    'seed':42
                }
                
                xgb_model = xgb.train(params, train_df, num_boost_round=5000, 
                          evals=[(train_df, 'train'), (valid_df, 'val')], 
                        early_stopping_rounds = 50, verbose_eval=100)
                joblib.dump(xgb_model, "save_model_st_joblib/xgb_st_" + str(self.seed) + '_' + str(idx) + '.ckpt')

    def model_load_infer_oof(self, model):
        self.model=model
        LABEL='survival_time'
        oof = np.zeros(len(self.train_data))
        skf = StratifiedKFold(n_splits=self.folds, random_state=self.seed, shuffle=True)

        for idx, (_, val_idx) in enumerate(skf.split(self.train_data, self.train_data[LABEL])):
            if self.model=='lgb':
                lgb_model = joblib.load("save_model_st_joblib/lgb_st_" + str(self.seed) + "_" + str(idx) + ".ckpt")
                oof[val_idx] = np.argmax(lgb_model.predict(self.train_data.loc[val_idx, self.features_]), axis=1)
            elif self.model=='rf':
                rf_model = joblib.load("save_model_st_joblib/rf_st_" + str(self.seed) + "_" + str(idx) + ".ckpt")
                oof[val_idx] = rf_model.predict(self.train_data.loc[val_idx, self.features_])
            elif self.model=='xgb':
                xgb_model = joblib.load("save_model_st_joblib/xgb_st_" + str(self.seed) + "_" + str(idx) + ".ckpt")
                oof[val_idx] = xgb_model.predict(xgb.DMatrix(self.train_data.loc[val_idx, self.features_]))
                
        else:
            oof = pd.concat([self.train_data, pd.DataFrame(oof, columns=['infer_survival_time'])], 1)
            oof = oof.loc[oof['week']==4, ['acc_id', 'survival_time', 'infer_survival_time']].reset_index(drop=True)
            self.oof = oof.copy()
            
            temp_dict = defaultdict()
            for true, idx in zip(self.true_index.keys(), self.true_index.values()):
                temp_dict[idx] = true
            else:    
                self.oof['survival_time'] = self.oof['survival_time'].apply(lambda x: temp_dict[x])
                self.oof['infer_survival_time'] = self.oof['infer_survival_time'].apply(lambda x: temp_dict[x])
                return self.oof
    
    def model_load_infer_pred(self):
        test1 = self.test1_data.loc[self.test1_data['week']==4].reset_index(drop=True)
        test2 = self.test2_data.loc[self.test2_data['week']==4].reset_index(drop=True)
        pred1 = np.zeros([len(test1), self.folds])
        pred2 = np.zeros([len(test2), self.folds])
        
        for idx in range(self.folds):
            if self.model=='lgb':
                lgb_model = joblib.load("save_model_st_joblib/lgb_st_" + str(self.seed) + "_" + str(idx) + ".ckpt")
                pred1[:, idx] = np.argmax(lgb_model.predict(test1[self.features_]), axis=1)
                pred2[:, idx] = np.argmax(lgb_model.predict(test2[self.features_]), axis=1)
            elif self.model=='rf':
                rf_model = joblib.load("save_model_st_joblib/rf_st_" + str(self.seed) + "_" + str(idx) + ".ckpt")
                pred1[:, idx] = rf_model.predict(test1[self.features_])
                pred2[:, idx] = rf_model.predict(test2[self.features_])
            elif self.model=='xgb':
                xgb_model = joblib.load("save_model_st_joblib/xgb_st_" + str(self.seed) + "_" + str(idx) + ".ckpt")
                pred1[:, idx] = xgb_model.predict(xgb.DMatrix(test1[self.features_]))
                pred2[:, idx] = xgb_model.predict(xgb.DMatrix(test2[self.features_]))
                
        else:
            test1 = pd.concat([test1['acc_id'], pd.DataFrame(pred1)], 1)
            test2 = pd.concat([test2['acc_id'], pd.DataFrame(pred2)], 1)
            
            temp_dict = defaultdict()
            for true, idx in zip(self.true_index.keys(), self.true_index.values()):
                temp_dict[idx] = true
            else:
                for i in range(5):
                    test1[i] = test1[i].apply(lambda x: temp_dict[x])
                    test2[i] = test2[i].apply(lambda x: temp_dict[x])
                else:
                    self.pred_test1 = test1
                    self.pred_test2 = test2    
                    return self.pred_test1, self.pred_test2
    
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
    PARAMS_ST = {
        'objective':'multiclass',
        'num_class':11,
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
        'seed':SEED
    }

    train = pd.read_csv('../preprocess/train.csv')
    test1 = pd.read_csv('../preprocess/test1.csv')
    test2 = pd.read_csv('../preprocess/test2.csv')

    train_label = pd.read_csv('../raw/train_label.csv')

    main_model = model(train, train_label, test1, test2, FOLDS, SEED)
    print('start main_model')

    # survivatl time
    main_model.labeling()
    main_model.train_fs(PARAMS_ST, 5000)

    # lgb
    select_model='lgb'
    main_model.train_st(PARAMS_ST, 5000, select_model)

    oof_st = main_model.model_load_infer_oof(select_model)
    oof_st.to_csv('../predict/oof_st_' + select_model + '.csv', index=False)
    pred_st1, pred_st2 = main_model.model_load_infer_pred()
    pred_st1.to_csv('../predict/pred_st1_' + select_model + '.csv', index=False)
    pred_st2.to_csv('../predict/pred_st2_' + select_model + '.csv', index=False)