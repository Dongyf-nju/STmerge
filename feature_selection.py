# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:57:09 2023
nohup python feature_selection.py > feature_selection.20260405.OPT-ST.log 2>&1 &
@author: veiga5
"""
import os
import netCDF4 as nc
import pandas as pd
import numpy as np
import numpy.ma as ma
import xarray as xr
import netCDF4 as nc
from netCDF4 import Dataset
import shapefile
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import time
import sys
import pickle

# 设置参数
OUTPUT_FOLDER = "default"
CAL_IMP_METHOD = "shap" # 计算特征重要性的方法，包括：perm、MDI、shap
feature_selection_strategy = "RFE"
SCALE_FACTOR = 0.01
NCPU = 30
SCORE="KGE"
SOIL_DEPTH = "10cm"
TARGET = f'OBS_ST_{SOIL_DEPTH}'
IF_OPT = True # 是否使用优化后的超参数
# TARGET = f'OBS_SKT'
BASIN_NAME = "CN"
K_FOLD = 6
# =============================================================================
print("====================================================")
print(f"OUTPUT_FOLDER: {OUTPUT_FOLDER}")
print(f"CAL_IMP_METHOD: {CAL_IMP_METHOD}")
print(f"feature_selection_strategy: {feature_selection_strategy}")
print(f"SCALE_FACTOR: {SCALE_FACTOR}")
print(f"NCPU: {NCPU}")
print(f"SCORE: {SCORE}")
print(f"SOIL_DEPTH: {SOIL_DEPTH}")
print(f"TARGET: {TARGET}")
print(f"K_FOLD: {K_FOLD}")
print("====================================================")
# =============================================================================
# 设置路径
BASE_PATH = "/home/yfdong/data/work/STmerge/v1.0/"
DATA_PATH = "/raid61/yfdong/data/work/STmerge/v1.0"

DB_PATH = os.path.join(DATA_PATH , "dataframe/database")

if TARGET == f'OBS_ST_{SOIL_DEPTH}':
    # ============================== ST =================================
    START_TEST_DATE = "2013-01-01"
    SAVE_PATH = os.path.join(DATA_PATH , "dataframe/train_output", SOIL_DEPTH, OUTPUT_FOLDER)
    FEATURE_PATH = os.path.join(SAVE_PATH, 'FeatureSelection')
    TRAIN_DATA_FILENAME = f"STmerge_daily_{TARGET}_1km_valid_by_Date_2013-01-01.parquet"
    VALIDATION_DATA_FILENAME = f"STmerge_daily_{TARGET}_1km_train_by_Date_2013-01-01.parquet"
    # TEST_DATA_FILENAME = f"STmerge_db_daily_test_by_Date.parquet"
    TARGET = f'OBS_ST_{SOIL_DEPTH}'
elif TARGET == f'OBS_SKT':
    # ============================== SKT =================================
    SOIL_DEPTH = "0cm"
    START_TEST_DATE = "2012-01-01"
    SAVE_PATH = os.path.join(DATA_PATH , "dataframe/train_output", SOIL_DEPTH, OUTPUT_FOLDER)
    FEATURE_PATH = os.path.join(SAVE_PATH, 'FeatureSelection')
    TRAIN_DATA_FILENAME = f'STmerge_daily_{TARGET}_1km_train_by_Date_{START_TEST_DATE}.parquet'
    VALIDATION_DATA_FILENAME = f'STmerge_daily_{TARGET}_1km_valid_by_Date_{START_TEST_DATE}.parquet'

STUDY_PATH = os.path.join(SAVE_PATH, 'Optuna')
print("SAVE_PATH:", SAVE_PATH)
print("Data Base Path:", DB_PATH)
print("FEATURE_PATH:", FEATURE_PATH)
print("STUDY_PATH:", STUDY_PATH)
print("TRAIN_DATA_FILENAME:", TRAIN_DATA_FILENAME)
print("VALIDATION_DATA_FILENAME:", VALIDATION_DATA_FILENAME)
print("TARGET:", TARGET)

# =============================================================================
def_features = pd.read_csv(os.path.join(DB_PATH, 'TotalFeatureColumns.csv'))["Feature"].tolist()
print(def_features)
import sys
sys.path.append(f"{BASE_PATH}/code/Library")
from MergeST import load_data, import_model_scaler, calc_feature_importances,plot_feature_importance,save_files,evaluate_features_serial, load_data_ByStratifiedSampling
Figure_path = os.path.join(SAVE_PATH, "FeatureSelection", "Fig") # 保存特征重要性的路径
SCORE_PATH = os.path.join(SAVE_PATH, "FeatureSelection", "Score") # 保存特征筛选过程评分的路径
full_data = load_data(TRAIN_DATA_FILENAME, DB_PATH)

train_data = load_data(TRAIN_DATA_FILENAME, DB_PATH)
print("full data size:", len(full_data), "train data size:", len(train_data), "resample ratio:",len(train_data)/len(full_data))

start = time.time()

def find_best_feature_set(BASIN_NAME, MODEL_NAME, INDEXthreshold, feature_score_df, all_features, score):
    """
    找到最优 score 对应的特征数量以及最优 score 值。
    
    参数：
    - feature_score_df: 包含特征数量和 score 评分的 DataFrame
    - threshold: 阈值，用于筛选有效的最优特征数量（默认为 0.001）
    
    返回：
    - min_score_x1: 最优特征数量
    - min_score_y1: 最优 score 值
    """
    # 找到最优 score 的索引
    if score=="RMSE":
        best_score_index = np.argmin(feature_score_df[f"Cross Validation Score:{score}"].values)
        best_score_x0 = feature_score_df["Number of Features"].values[best_score_index]
        best_score_y0 = feature_score_df[f"Cross Validation Score:{score}"].values[best_score_index]
        threshold =best_score_y0*INDEXthreshold
    elif score=="KGE":
        best_score_index = np.argmax(feature_score_df[f"Cross Validation Score:{score}"].values)
        best_score_x0 = feature_score_df["Number of Features"].values[best_score_index]
        best_score_y0 = feature_score_df[f"Cross Validation Score:{score}"].values[best_score_index]
        threshold =best_score_y0*INDEXthreshold
    best_score_y1 = best_score_y0
    best_score_x1 = best_score_x0
    print("ORI", best_score_y1, best_score_x1)
    # 找到与最优 score 之差小于阈值的最优特征数量
    for index in np.arange(0,best_score_index+1,1):
        print(index)
        temp_rmse_y = feature_score_df[f"Cross Validation Score:{score}"].values[index]
        print(temp_rmse_y, best_score_y0 ,threshold)
        if abs(temp_rmse_y - best_score_y0) < threshold:
            best_score_y1 = temp_rmse_y
            best_score_x1 = feature_score_df["Number of Features"].values[index]
            print("Number of Features",best_score_x1)
            break
    #print(best_score_x1)
    BASINlist.append(BASIN_NAME)
    MODELlist.append(MODEL_NAME)
    MinFeature.append(best_score_x1)
    BestScore.append(best_score_y1)
    FS_list = {
        "MODEL":MODELlist,
        "BASIN":BASINlist,
        f"Best features: ":MinFeature,
        f"Best Score: ":BestScore
        }
    print(FS_list)
    subset_feature = all_features[:best_score_x1]
    print(all_features)
    return FS_list ,subset_feature

def get_feature(feature_selection_strategy):
    if feature_selection_strategy =="HRFE":
        #HRFE
        print(f"feature_selection_strategy: {feature_selection_strategy}")
        all_features = pd.read_excel(os.path.join(FEATURE_PATH ,f"{CAL_IMP_METHOD}_total_imp_df.xlsx"))["Feature"].tolist()  
    elif feature_selection_strategy =="RFE":
        if CAL_IMP_METHOD=="shap":
            print(f"Feature selection strategy: {feature_selection_strategy}",f"Calculate feature importance by: {CAL_IMP_METHOD}")
            all_features = pd.read_excel(os.path.join(FEATURE_PATH ,f"{CAL_IMP_METHOD}_{MODEL_NAME}_{TARGET}_OPT-False_FS-{IF_FS}_importance_df.xlsx"))["Feature"].tolist() 
        # RFE
        elif CAL_IMP_METHOD=="MDI":
            print(f"Feature selection strategy: {feature_selection_strategy}",f"Calculate feature importance by: {CAL_IMP_METHOD}")
            all_features = pd.read_excel(os.path.join(FEATURE_PATH ,f"MDI_{MODEL_NAME}_OPT-{IF_OPT}_FS-{IF_FS}_importance_df.xlsx"))["Feature"].tolist() 
    return all_features
# =============================================================================
# feature_selection_strategy = "RFE"
# Constants
MODELS = ["CB"]
SCORE = 'KGE'
SCALE_FACTOR = 0.01
NCPU = -1
INDEXthreshold = 0.001
BASINlist = []
MODELlist = []
MinFeature = []
BestScore = []
study = False
IF_FS = False # 是否使用最优特征子集
for feature_selection_strategy in ["RFE"]:
    FS_strategy_df = pd.DataFrame()
    feature_score_df=pd.DataFrame()
    print(f"*******************{BASIN_NAME}*******************")
    for i, MODEL_NAME in enumerate(MODELS):
        print(f"#----------------{MODEL_NAME}---------------#")
        # 如果使用优化后的超参数，加载 Optuna 的 study 对象
        if IF_OPT == True:
            STUDY_FILENAME = f'{BASIN_NAME}_{MODEL_NAME}_{TARGET}_{feature_selection_strategy}_shap_study.pkl'
            with open(os.path.join(STUDY_PATH, STUDY_FILENAME), 'rb') as f:
                study = pickle.load(f)
        print(SAVE_PATH)
        # 读取数据库
        # 获取特征
        all_features = get_feature(feature_selection_strategy)
        all_columns = ['ID',  "Date" , TARGET] + all_features
        db_df = train_data.reindex(columns = all_columns)
        basic_columns = 3 #开始迭代的特征数量
        # -----------------------------评估不同特征子集的表现-----------------------
        feature_list = evaluate_features_serial(MODEL_NAME, BASIN_NAME,  study, IF_OPT, train_data, all_features, basic_columns, target=TARGET, score =SCORE, scale_factor=SCALE_FACTOR, kf=K_FOLD , n_cpu =NCPU)
        MODEL_feature_score_df = pd.DataFrame(feature_list)
        feature_score_df =  pd.concat([feature_score_df, MODEL_feature_score_df], ignore_index=True)
        print(feature_score_df)
        # ------------------------------找到最优值点的坐标-------------------------
        FS_list ,subset_feature= find_best_feature_set(BASIN_NAME, MODEL_NAME,INDEXthreshold, feature_score_df, all_features,score =SCORE)
        # ---------------------添加到列表--------------------------------
        FS_strategy_df = pd.DataFrame(FS_list)
        print(pd.DataFrame(FS_list))
        subset_feature_df = pd.DataFrame(subset_feature, columns=['Feature'])
        print(subset_feature_df)
        save_files(FEATURE_PATH, f"IF_OPT-{IF_OPT}_{feature_selection_strategy}_{CAL_IMP_METHOD}_{MODEL_NAME}_{BASIN_NAME}_{TARGET}_subset_feature.K_FOLD-{K_FOLD}.csv", subset_feature_df)
        save_files(FEATURE_PATH, f"IF_OPT-{IF_OPT}_{feature_selection_strategy}_{CAL_IMP_METHOD}_{MODEL_NAME}_{BASIN_NAME}_{TARGET}_feature_score.K_FOLD-{K_FOLD}.csv", feature_score_df)
        save_files(FEATURE_PATH, f"IF_OPT-{IF_OPT}_{feature_selection_strategy}_{CAL_IMP_METHOD}_{MODEL_NAME}_{BASIN_NAME}_{TARGET}_strategy_score.K_FOLD-{K_FOLD}.csv", FS_strategy_df)

end = time.time()
print(f"Elapse Time: {end - start}Seconds")
