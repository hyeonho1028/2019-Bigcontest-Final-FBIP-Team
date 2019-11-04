## fbip team

### Path

```bash
.
├── raw
│   ├── train_activity.csv
│   ├── ...
│   ├── test1_activity.csv
│   ├── ...
│   ├── test2_activity.csv
│   └── train_label.csv
├── preprocess
│   ├── train.csv
│   ├── test1.csv
│   ├── test2.csv
│   └── preprocess.py
├── model
│   ├── save_model_joblib
│   │   ├── pktl
│   │   ├── pktl
│   │   └── pktl
│   ├── train.csv
│   └── preprocess.py
├── metric
│   └── score_function.py
├── predict
│   ├── test1_predict.csv
│   └── test2_predict.csv
└── README.md
```

### Version

```bash
base
python==3.6.8
joblib==0.13.2

handing
pandas==0.24.2
numpy==1.16.5
imblearn==0.5.0

model
lightgbm==2.2.3
xgboost==0.90
```

### Execute

```bash
directory
./fbip_team/preprocess/
$> python data_transform.py

directory
./fbip_team/model/
$> python survival_time.py
$> python total_amount_spent.py
```

### Description

##### data_transform.py

```python
# 각 데이터셋을 변환시키는 클래스
class data_transform

# inner function
# week를 생성
def create_week

# 각 테이블을 week기준으로 변환, 집계 변수 생성
def activity_transform
def payment_transfrom
def trade_transform
def combat_transform
def pledge_transform

# data merge
# playstyle 변수 생성
# PCA 변수 생성
# 진입시기 변수 생성
```

##### survival_time.py

```python
# survival time을 y로 모델 학습하는 클래스
class model

# inner function
# weekly data transform을 위한 week labeling
def labeling

# feature selection을 위한 함수
# 각 survival 카테고리 별 undersampling 후, lgb model의 feature imoortace를 기준으로 변수 선택
def train_fs

# survival time의 학습을 위한 함수
# 각 survival 카테고리 별 undersampling 후, 5-fold train
# 가용 모델 : lgb
def train_st

# out of fold를 통해 train inference 도출
def model_load_infer_oof

# test 데이터 적용을 통해 predict 값 도출
# 이 때, 4주차의 데이터만을 사용
def model_load_infer_pred

# 원하는 값을 return
# 가능 값 : train, test1, test2, model_st, model_tas, true_dict, feature
def load
```

##### total_amount_spent.py

```python
# total amount spent를 y로 모델 학습하는 클래스
class model

# inner function
# weekly data transform을 위한 week labeling
# payment와 survival time을 고려한 total_amount_spent 재정의
def labeling_tas
    def temp_func

# feature selection을 위한 함수
# 각 소수점 첫째자리 기준 undersampling 후, lgb model의 feature imoortace를 기준으로 변수 선택
def train_fs

# total_amount_spent의 학습을 위한 함수
# 소수점 첫째자리 기준 undersampling 후, 5-fold train
# 가용 모델 : lgb
def train_tas

# out of fold를 통해 train inference 도출
def model_load_infer_oof

# test 데이터 적용을 통해 predict 값 도출
# 이 때, 4주차의 데이터만을 사용
def model_load_infer_pred

# 원하는 값을 return
# 가능 값 : train, test1, test2, model_st, model_tas, true_dict, feature
def load
```



