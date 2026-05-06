#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:57:09 2023
nohup python feature_selection.py > feature_selection.20260322_SITE_FRAC.log 2>&1 &
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
NCPU = 30
SCORE="KGE"
BASIN_NAME = "CN"
# =============================================================================
# 设置路径
BASE_PATH = "/home/yfdong/data/work/STmerge/v1.0/"
DATA_PATH = "/raid61/yfdong/data/work/STmerge/v1.0"
DB_PATH = os.path.join(DATA_PATH , "dataframe/database")

# ============================== ST =================================
SOIL_DEPTH = "10cm"
START_TEST_DATE = "2013-01-01"
TARGET = f'OBS_ST_{SOIL_DEPTH}'
SAVE_PATH = os.path.join(DATA_PATH , "dataframe/train_output", SOIL_DEPTH, OUTPUT_FOLDER)
FEATURE_PATH = os.path.join(SAVE_PATH, 'FeatureSelection')
TRAIN_DATA_FILENAME = f"STmerge_daily_{TARGET}_1km_train_by_Date_{START_TEST_DATE}.parquet"
VALIDATION_DATA_FILENAME = f"STmerge_daily_{TARGET}_1km_valid_by_Date_{START_TEST_DATE}.parquet"

# # ============================== SKT =================================
# SOIL_DEPTH = "0cm"
# START_TEST_DATE = "2012-01-01"
# SAVE_PATH = os.path.join(DATA_PATH , "dataframe/train_output", SOIL_DEPTH, OUTPUT_FOLDER)
# FEATURE_PATH = os.path.join(SAVE_PATH, 'FeatureSelection')
# TRAIN_DATA_FILENAME = f'SKTmerge_daily_10cm_1km_train_by_Date_{START_TEST_DATE}.parquet'
# VALIDATION_DATA_FILENAME = f'SKTmerge_daily_10cm_1km_valid_by_Date_{START_TEST_DATE}.parquet'
# TARGET = f'OBS_SKT'

def_features = pd.read_csv(os.path.join(DB_PATH, 'TotalFeatureColumns.csv'))["Feature"].tolist()
print(def_features)
import sys
sys.path.append(f"{BASE_PATH}/code/Library")
from MergeST import load_data, import_model_scaler, calc_feature_importances,plot_feature_importance,save_files,evaluate_features_serial, load_data_ByStratifiedSampling
Figure_path = os.path.join(SAVE_PATH, "FeatureSelection", "Fig") # 保存特征重要性的路径
SCORE_PATH = os.path.join(SAVE_PATH, "FeatureSelection", "Score") # 保存特征筛选过程评分的路径
# 读取数据
full_data = load_data(TRAIN_DATA_FILENAME, DB_PATH)
train_data = load_data(TRAIN_DATA_FILENAME, DB_PATH, random_state = 42)
print("full data size:", len(full_data), "train data size:", len(train_data), "resample ratio:",len(train_data)/len(full_data))
# ================================ 计算特征重要性排名 ================================ 
# 过滤所有警告
# warnings.filterwarnings("ignore")

# =============================================================================
start = time.time()
# 设置专属参数
MODELS = ["CB"]
metric_dfs_1 = pd.DataFrame()
NCPU = 24
IF_OPT = False # 是否使用优化后的超参数
IF_FS = False # 是否使用最优特征子集


predicted_data_dfs = pd.DataFrame()
metric_dfs_2 = pd.DataFrame()
total_imp_df = pd.DataFrame()
merge_imp_df = pd.DataFrame()

for MODEL_NAME in MODELS:
    print(f"#----------------{MODEL_NAME}---------------#")
    # 特征重要性排序文件
    IMPORTANCE_NAME = f"{CAL_IMP_METHOD}_{MODEL_NAME}_{TARGET}_OPT-{IF_OPT}_FS-{IF_FS}_importance_df.xlsx"
    print(IMPORTANCE_NAME)
    # ------------------------导入训练好的模型和标准化方式---------------
    if IF_OPT == True:
        
        if IF_FS ==True: # 优化后的超参数和最优特征子集 ->用于构建最终的融合模型
            model, scaler = import_model_scaler(os.path.join(SAVE_PATH, "model"),  f"{MODEL_NAME}_{TARGET}_shap_OPT")
            FEATURE_FILENAME = f"{feature_selection_strategy}_{CAL_IMP_METHOD}_{MODEL_NAME}_{BASIN_NAME}_{IF_OPT}_subset_feature.csv"
            features = pd.read_csv(os.path.join(FEATURE_PATH, "csv", FEATURE_FILENAME))["Feature"].tolist()
        else: # 优化后的超参数和全部的特征 ->和默认的超参数，全部的特征对比评估：超参数优化对于特征重要性排序的影响
            features = def_features.copy()
            model, scaler = import_model_scaler(os.path.join(SAVE_PATH, "model"),  f"{MODEL_NAME}_{TARGET}_shap_OPT_FS{IF_FS}")
    elif IF_OPT ==False: # 使用默认的超参数和全部的特征
        model, scaler = import_model_scaler(os.path.join(SAVE_PATH, "model"),  f"{MODEL_NAME}_{TARGET}_DEF_allFeature")
        features = def_features.copy()
    print(features)
    # -----------------------------------特征选择---------------------------------
    # 提取特征和目标变量
    X_test = train_data[features]
    y_test = train_data[TARGET]
    # 对测试集进行标准化操作（使用训练集的统计信息）
    X_test_scaled = scaler.transform(X_test)
    # ------------------------计算特征重要性排名---------------------------
    # 定义路径
    shap_df, importance_df = calc_feature_importances(CAL_IMP_METHOD , model, X_test_scaled, y_test, features, n_jobs=NCPU)
    importance_df['NIM'] = (importance_df['Importance'] - importance_df['Importance'].min()) / (importance_df['Importance'].max() - importance_df['Importance'].min())
    importance_df = importance_df.reset_index(drop=False)
    print(importance_df)
    # 保存文件
    save_files(os.path.join(SAVE_PATH, "FeatureSelection"), IMPORTANCE_NAME, importance_df)   
    # 读取文件
    importance_df_file = os.path.join(os.path.join(SAVE_PATH,"FeatureSelection"), IMPORTANCE_NAME)
    importance_df = pd.read_excel(importance_df_file)
    # 绘图并保存
    filename =f"{MODEL_NAME}_feature_importance.png"
    title = f'{MODEL_NAME}'
    plot_feature_importance(importance_df, title, dpi=300, path = Figure_path ,filename = filename)
    # ------------------------计算特征重要性排名---------------------------
    total_imp_df[MODEL_NAME] = importance_df["Feature"]
    merge_imp_df = pd.concat([merge_imp_df , importance_df],axis=0)
# =============================================================================
# 根据归一化的特征重要性求和来筛选特征重要性
total_imp_df_2 = merge_imp_df.groupby("Feature").sum().reset_index()
total_imp_df_2 = total_imp_df_2.sort_values(by='NIM',ascending= False)

save_files(os.path.join(SAVE_PATH,"FeatureSelection"),f"{MODEL_NAME}_{CAL_IMP_METHOD}_OPT{IF_OPT}_FS{IF_FS}_{TARGET}_shap_values_df.parquet", shap_df)     

end = time.time()
print(f"Elapse Time: {end - start}Seconds")
print("Save path: ",importance_df_file)
