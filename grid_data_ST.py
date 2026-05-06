#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:57:09 2023
nohup python grid_data_ST.py > grid_data_ST.20260320-2.log 2>&1 &
@author: veiga5
"""
import numpy as np
import netCDF4 as nc
from netCDF4 import Dataset
import os
import pandas as pd
import time
import calendar
from datetime import datetime, timedelta
import os
import numpy as np
from netCDF4 import Dataset as ncDataset
start = time.time()

# =============================================================================
# 
def get_previous_date_str(date_str):
    # 解析 date_str 为 datetime 对象
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    date = datetime(year, month, day)
    # 计算三天前的日期
    PV3Ddate = date - timedelta(days=3)
    PV7Ddate = date - timedelta(days=7)
    # 格式化三天前的日期为字符串
    PV3Ddate_str = f"{PV3Ddate.year}{str(PV3Ddate.month).zfill(2)}{str(PV3Ddate.day).zfill(2)}"
    PV7Ddate_str = f"{PV7Ddate.year}{str(PV7Ddate.month).zfill(2)}{str(PV7Ddate.day).zfill(2)}"
    return PV3Ddate_str, PV7Ddate_str

def abbre_to_full(abbreviation):
    # 定义缩写到全称的映射字典
    abbreviation_mapping = {
        "RF": "RandomForest",
        "LGBM": "LightGBM",
        "CB": "CatBoost",
        "XGB": "XGBoost",
        # "WHO": "World Health Organization"
    }
    # 查找缩写对应的全称
    full_name = abbreviation_mapping.get(abbreviation.upper())
    if full_name:
        return full_name
    else:
        return f"未找到 '{abbreviation}' 的全称。"
    
def create_doy_cos_2d(date_str, shape, period=None):
    """
    根据日期字符串生成 DOY_cos 的二维数组（所有像元值相同）。
    Parameters:
    ----------
    date_str : str
        日期字符串，格式为 'YYYYMMDD'，如 '20230115'
    shape : tuple of int
        输出数组的形状，如 (lat_size, lon_size)
    period : int or None
        周期天数。若为 None，则根据是否闰年自动选择 366 或 365
    Returns:
    -------
    doy_cos_2d : np.ndarray
        形状为 `shape` 的 float32 数组，值为 -cos(2π * DOY / period)
    """
    # 解析日期
    dt = datetime.strptime(date_str, "%Y%m%d")
    doy = dt.timetuple().tm_yday  # 年积日（1–366）
    # 自动确定周期
    if period is None:
        period = 366 if calendar.isleap(dt.year) else 365
    # 计算 DOY_cos 标量值
    doy_cos_value = -np.cos(2 * np.pi * doy / period)
    # 扩展为二维数组
    doy_cos_2d = np.full(shape, doy_cos_value, dtype=np.float32)
    return doy_cos_2d

from netCDF4 import date2num
def create_nc_file(date, XLAT, XLON, SM_Daily, GRIDDATA_FILEPATH, MODEL_NAME, SOIL_DEPTH):
    """
    创建NetCDF文件，用于存储土壤温度数据。
    
    参数:
    - date: datetime对象，表示数据日期。
    - XLAT: numpy数组，表示纬度数据。
    - XLON: numpy数组，表示经度数据。
    - SM_Daily: numpy二维数组，表示土壤温度数据。
    - GRIDDATA_FILEPATH: 字符串，表示文件存储路径。
    - MODEL_NAME: 字符串，表示模型名称。
    - SOIL_DEPTH: 字符串，表示土壤深度（"10cm"或"40cm"）。
    """
    # 显示转化为np.float32
    SM_Daily = SM_Daily.astype(np.float32)
    # 获取原始二维数组的形状
    lat_size, lon_size = XLAT.shape[0], XLON.shape[0]
    # 创建NetCDF文件
    with nc.Dataset(GRIDDATA_FILEPATH, 'w', format='NETCDF4') as nc_file:
        # 定义维度
        nc_file.createDimension('time', None)  # 无限制时间维度
        nc_file.createDimension('lat', lat_size)
        nc_file.createDimension('lon', lon_size)
        # 创建变量
        time_var = nc_file.createVariable('time', np.float64, ('time',))
        lat_var = nc_file.createVariable('lat', np.float32, ('lat',))
        lon_var = nc_file.createVariable('lon', np.float32, ('lon',))
        save_var = nc_file.createVariable('ST', np.float32, ('lat', 'lon'), zlib=True)  # 开启压缩
        # 设置变量属性
        time_var.units = 'days since 1900-01-01'
        time_var.calendar = 'gregorian'
        lat_var.units = 'degrees_north'
        lon_var.units = 'degrees_east'
        save_var.units = 'K'
        save_var.long_name = f'Soil temperature at {SOIL_DEPTH}'
        # 写入数据
        time_var[:] = date2num(date, units=time_var.units, calendar=time_var.calendar)
        lat_var[:] = XLAT[:]  
        lon_var[:] = XLON[:]  
        save_var[:, :] = SM_Daily 
        #save_var[:, :] = np.round(SM_Daily.astype('f4'), 4)  # 强制单精度+四舍五入
        MODEL_Full_NAME = abbre_to_full(MODEL_NAME)
        # 添加全局属性
        nc_file.description = f'{SOIL_DEPTH} cm Multi-source Soil Temperature Fusion Data Based on CatBoost model.'# The predictive features of integrated data include: {features}'
        nc_file.history = f'File created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    pass

# 预加载目标网格数据，提高效率
def prepare_target_grid(target_file):
    dst_nc = Dataset(target_file)
    dst_lat = dst_nc.variables['lat'][:]
    dst_lon = dst_nc.variables['lon'][:]
    mask_dst = dst_nc.variables['Band1'][:]
    #mask_dst[mask_dst <= 0] = np.nan
    mask_dst[mask_dst > 0] = 1
    dst_nc.close()
    return dst_lat, dst_lon, mask_dst

import xarray as xr
def read_nc_variable(nc_file , variable_name):
    with nc.Dataset(nc_file) as ds:
        ori_var = ds.variables[variable_name][:]  
        ori_lat = ds.variables['lat'][:]
        ori_lon = ds.variables['lon'][:]
    ori_var[(abs(ori_var) > 1e9)] = np.nan
    data = xr.DataArray(ori_var, coords=[ori_lat, ori_lon], dims=['lat', 'lon'])
    data_filled = data.interpolate_na(dim='lat', method='nearest').interpolate_na(dim='lon', method='nearest')
    return data_filled.values

def Grid_data_for_year(year, START_MONTH, END_MONTH, SOIL_DEPTH, MODEL_NAME, DSTresolution):
    BASE_PATH = "/home/yfdong/data/work/STmerge/v1.0"
    DATA_PATH = "/raid61/yfdong/data/work/STmerge/v1.0"
    SAVE_PATH = os.path.join(DATA_PATH , "dataframe/train_output", SOIL_DEPTH, OUTPUT_FOLDER)
    FEATURE_PATH = os.path.join(SAVE_PATH, 'FeatureSelection')
    import sys
    sys.path.append(f"{BASE_PATH}/code/Library")
    from MergeST import import_model_scaler 
    # 设置参数
    Spatial_Resolution = "1km"
    CAL_IMP_METHOD = "shap"
    TARGET = f'OBS_ST_{SOIL_DEPTH}'
    MODEL_PATH = os.path.join(DATA_PATH, f"dataframe/train_output/{SOIL_DEPTH}/default/model") # 模型路径
    FeatureData_path = os.path.join(DATA_PATH, "grid/feature/dst/") # 特征数据路径
    GRIDDATA_PATH = os.path.join(DATA_PATH, "GridData", "ori", DSTresolution, SOIL_DEPTH, str(year)) # 输出路径
    print(f"Model Path: {MODEL_PATH}")
    print(f"Feature Data Path: {FeatureData_path}")
    print(f"Grid Data Output Path: {GRIDDATA_PATH}")
    # 导入模型和标准化器,以及对应的特征列表
    if year >= 2000:
        model, scaler = import_model_scaler(MODEL_PATH, f"CB_shap_IF_OPT-True_FS-True_Total.2000-2024")
        features = pd.read_csv(os.path.join(FEATURE_PATH, f'RFE_shap_CB_CN_{TARGET}_subset_feature.2000-2024.csv'))["Feature"].tolist()
    elif year >= 1980 and year < 2000:
        model, scaler = import_model_scaler(MODEL_PATH, f"CB_shap_IF_OPT-True_FS-True_Total.1980-1999")
        features = pd.read_csv(os.path.join(FEATURE_PATH, f'RFE_shap_CB_CN_{TARGET}_subset_feature.1980-1999.csv'))["Feature"].tolist()
    elif year >= 1960 and year < 1980:
        model, scaler = import_model_scaler(MODEL_PATH, f"CB_shap_IF_OPT-True_FS-True_Total.1960-1979")
        features = pd.read_csv(os.path.join(FEATURE_PATH, f'RFE_shap_CB_CN_{TARGET}_subset_feature.1960-1979.csv'))["Feature"].tolist()
    else:
        print("Year out of range.")
    print('features:', features)
    print('features nums', len(features))
    # =============================================================================
    # 读取MASK文件
    target_file = f"/raid61/yfdong/data/StaticData/ChinaMask/China_mask_{DSTresolution}.nc"
    lat, lon, mask_data = prepare_target_grid(target_file)
    mask = mask_data.copy()

    # 读取静态变量
    DEM_file = os.path.join(FeatureData_path ,Spatial_Resolution , "Static", f"HydroSHEDS.DEM.{DSTresolution}.nc")
    # SLOPE_file = os.path.join(FeatureData_path ,Spatial_Resolution , "Static", f"HydroSHEDS.SLOPE.{DSTresolution}.nc")
    Soil_bd_file = os.path.join(FeatureData_path ,Spatial_Resolution , "Static", f"ISSCAS.bd.05_{DSTresolution}.nc")
    Soil_thickness_file = os.path.join(FeatureData_path ,Spatial_Resolution , "Static", f"ISSCAS.thickness.05_{DSTresolution}.nc")
    # Soil_ph_file = os.path.join(FeatureData_path ,Spatial_Resolution , "Static", f"ISSCAS.ph.05_{DSTresolution}.nc")
    Soil_soc_file = os.path.join(FeatureData_path ,Spatial_Resolution , "Static", f"ISSCAS.soc.05_{DSTresolution}.nc")
    Soil_cec_file = os.path.join(FeatureData_path ,Spatial_Resolution , "Static", f"ISSCAS.cec.05_{DSTresolution}.nc")
    # Soil_tp_file = os.path.join(FeatureData_path ,Spatial_Resolution , "Static", f"ISSCAS.tp.05_{DSTresolution}.nc")
    # Soil_tk_file = os.path.join(FeatureData_path ,Spatial_Resolution , "Static", f"ISSCAS.tk.05_{DSTresolution}.nc")
    Soil_tn_file = os.path.join(FeatureData_path ,Spatial_Resolution , "Static", f"ISSCAS.tn.05_{DSTresolution}.nc")
    # Soil_cf_file = os.path.join(FeatureData_path ,Spatial_Resolution , "Static", f"ISSCAS.cf.05_{DSTresolution}.nc")
    # Soil_btsnd_file = os.path.join(FeatureData_path ,Spatial_Resolution , "Static", f"ISSCAS.btsnd.05_{DSTresolution}.nc")
    # Soil_btslt_file = os.path.join(FeatureData_path ,Spatial_Resolution , "Static", f"ISSCAS.btslt.05_{DSTresolution}.nc")
    # Soil_btcly_file = os.path.join(FeatureData_path ,Spatial_Resolution , "Static", f"ISSCAS.btcly.05_{DSTresolution}.nc")
    # Soil_texcls_file = os.path.join(FeatureData_path ,Spatial_Resolution , "Static", f"ISSCAS.texcls.05_{DSTresolution}.nc")

    DEM = read_nc_variable(DEM_file , 'DEM')
    # SLOPE = read_nc_variable(SLOPE_file , 'SLOPE')
    BD = read_nc_variable(Soil_bd_file , 'bd')
    Thickness = read_nc_variable(Soil_thickness_file , 'thickness')
    # PH = read_nc_variable(Soil_ph_file , 'ph')
    SOC = read_nc_variable(Soil_soc_file , 'soc')
    CEC = read_nc_variable(Soil_cec_file , 'cec')
    # TP = read_nc_variable(Soil_tp_file , 'tp')
    # TK = read_nc_variable(Soil_tk_file , 'tk')
    TN = read_nc_variable(Soil_tn_file , 'tn')
    # CF = read_nc_variable(Soil_cf_file , 'cf')
    # Sand = read_nc_variable(Soil_btsnd_file , 'btsnd')
    # Silt = read_nc_variable(Soil_btslt_file , 'btslt')
    # Clay = read_nc_variable(Soil_btcly_file , 'btcly')
    # Texture = read_nc_variable(Soil_texcls_file , 'texcls')

    # 设置column
    cols_ST = ['ERA5_ST',  'GLDAS_Noah_ST',  'MERRA2_ST', 'JRA3Q_ST',
               'ERA5_SKT', 'GLDAS_Noah_SKT',  'ERA5_Land_SKT', 'MERRA2_SKT', 'JRA3Q_SKT'] #'ERA5_Land_ST', 

    
    os.makedirs(GRIDDATA_PATH, exist_ok=True)
    for month in range(START_MONTH,END_MONTH+1):
        _, monthRange = calendar.monthrange(year,month)
        for day in range(1, monthRange+1):
        # for day in range(1, 2):
            date_str = f"{year}{str(month).zfill(2)}{str(day).zfill(2)}"
            PV3Ddate_str, PV7Ddate_str = get_previous_date_str(date_str)
            # 验证输入参数
            if SOIL_DEPTH not in ["0cm", "10cm", "40cm"]:
                raise ValueError("Invalid SOIL_DEPTH. Expected '10cm' or '40cm'..nc.nc")
            depth_to_filename = {
                "0cm": f"{date_str}.CSTX_000000.nc",
                "10cm": f"{date_str}.CSTX_000010.nc",
                "40cm": f"{date_str}.CSTX_010040.nc",
            }
            GRIDDATA_FILENAME = depth_to_filename[SOIL_DEPTH]
            GRIDDATA_FILEPATH = os.path.join(GRIDDATA_PATH, GRIDDATA_FILENAME)
            # 判断文件是否存在
            if os.path.exists(GRIDDATA_FILEPATH):
                continue
            start_merge = time.time()
            # ================================= 读取格点数据 =======================================
            # 读取动态变量（Daily）
            # 地表温度
            # GLDAS_Noah_SKT_file = os.path.join(FeatureData_path ,Spatial_Resolution , 'daily', "GLDAS_Noah_SKT", f"{date_str}.GLDAS_Noah_SKT.nc")
            # GLDAS_CLSM_SKT_file = os.path.join(FeatureData_path ,Spatial_Resolution , 'daily', "GLDAS_CLSM_SKT", f"{date_str}.GLDAS_CLSM_SKT.nc")
            # ERA5_Land_SKT_file = os.path.join(FeatureData_path ,Spatial_Resolution , 'daily', "ERA5_Land_SKT", f"{date_str}.ERA5_Land_SKT.nc")
            ERA5_SKT_file = os.path.join(FeatureData_path ,Spatial_Resolution , 'daily', "ERA5_SKT", f"{date_str}.ERA5_SKT.nc")
            MERRA2_SKT_file = os.path.join(FeatureData_path ,Spatial_Resolution , 'daily', "MERRA2_SKT", f"{date_str}.MERRA2_SKT.nc") # 1980~2025
            # JRA3Q_SKT_file = os.path.join(FeatureData_path ,Spatial_Resolution , 'daily', "JRA3Q_SKT", f"{date_str}.JRA3Q_SKT.nc")
            # 土壤温度
            GLDAS_Noah_ST_file = os.path.join(FeatureData_path ,Spatial_Resolution , 'daily', "GLDAS_Noah_ST", f"{date_str}.GLDAS_Noah_ST.nc")
            ERA5_Land_ST_file = os.path.join(FeatureData_path ,Spatial_Resolution , 'daily', "ERA5_Land_ST", f"{date_str}.ERA5_Land_ST.nc")
            ERA5_ST_file = os.path.join(FeatureData_path ,Spatial_Resolution , 'daily', "ERA5_ST", f"{date_str}.ERA5_ST.nc")
            MERRA2_ST_file = os.path.join(FeatureData_path ,Spatial_Resolution , 'daily', "MERRA2_ST", f"{date_str}.MERRA2_ST.nc") # 1980~2025
            JRA3Q_ST_file = os.path.join(FeatureData_path ,Spatial_Resolution , 'daily', "JRA3Q_ST", f"{date_str}.JRA3Q_ST.nc")
            # 土壤温度
            ERA5_Land_SM_file = os.path.join(FeatureData_path ,Spatial_Resolution , 'daily', "ERA5_Land_SM", f"{date_str}.ERA5_Land_SM.nc")
            # 滞后变量
            # ERA5_ST_lag3_file = os.path.join(FeatureData_path ,Spatial_Resolution , 'daily', "ERA5_ST", f"{PV3Ddate_str}.ERA5_ST.nc")
            ERA5_Land_ST_lag3_file = os.path.join(FeatureData_path ,Spatial_Resolution , 'daily', "ERA5_Land_ST", f"{PV3Ddate_str}.ERA5_Land_ST.nc")
            ERA5_Land_ST_lag7_file = os.path.join(FeatureData_path ,Spatial_Resolution , 'daily', "ERA5_Land_ST", f"{PV7Ddate_str}.ERA5_Land_ST.nc")
            MERRA2_ST_lag7_file = os.path.join(FeatureData_path ,Spatial_Resolution , 'daily', "MERRA2_ST", f"{PV7Ddate_str}.MERRA2_ST.nc") # 1980~2025
            # ERA5_Land_SM_lag7_file = os.path.join(FeatureData_path ,Spatial_Resolution , 'daily', "ERA5_Land_SM", f"{PV7Ddate_str}.ERA5_Land_SM.nc")
            # 气象变量
            # Prec_file = os.path.join( FeatureData_path ,Spatial_Resolution , 'daily', "Prec",  f"{date_str}.Prec.nc")
            SUBRF_file = os.path.join( FeatureData_path ,Spatial_Resolution , 'daily', "SUBRF",  f"{date_str}.SUBRF.nc")   
            # SW_Down_file = os.path.join( FeatureData_path ,Spatial_Resolution , 'daily', "SW_Down",  f"{date_str}.SW_Down.nc")
            # LW_Down_file = os.path.join( FeatureData_path ,Spatial_Resolution , 'daily', "LW_Down",  f"{date_str}.LW_Down.nc")
            SP_file = os.path.join( FeatureData_path ,Spatial_Resolution , 'daily', "SP",  f"{date_str}.SP.nc")
            T2m_file = os.path.join( FeatureData_path ,Spatial_Resolution , 'daily', "T2m",  f"{date_str}.T2m.nc")
            # Td_file = os.path.join( FeatureData_path ,Spatial_Resolution , 'daily', "Td",  f"{date_str}.Td.nc")
            # ET_file = os.path.join( FeatureData_path ,Spatial_Resolution , 'daily', "ET",  f"{date_str}.ET.nc")
            NLW_file = os.path.join( FeatureData_path ,Spatial_Resolution , 'daily', "NLW",  f"{date_str}.NLW.nc")
            # NSW_file = os.path.join( FeatureData_path ,Spatial_Resolution , 'daily', "NSW",  f"{date_str}.NSW.nc")
            # LH_file = os.path.join( FeatureData_path ,Spatial_Resolution , 'daily', "LH",  f"{date_str}.LH.nc")
            # 读取变量
            # GLDAS_Noah_SKT = read_nc_variable(GLDAS_Noah_SKT_file , 'GLDAS_Noah_SKT')
            # GLDAS_CLSM_SKT = read_nc_variable(GLDAS_CLSM_SKT_file , 'GLDAS_CLSM_SKT')
            # ERA5_Land_SKT = read_nc_variable(ERA5_Land_SKT_file , 'ERA5_Land_SKT')
            ERA5_SKT = read_nc_variable(ERA5_SKT_file , 'ERA5_SKT')
            MERRA2_SKT = read_nc_variable(MERRA2_SKT_file , 'MERRA2_SKT') # 1980~2025
            # JRA3Q_SKT = read_nc_variable(JRA3Q_SKT_file , 'JRA3Q_SKT')
            GLDAS_Noah_ST = read_nc_variable(GLDAS_Noah_ST_file , 'GLDAS_Noah_ST')
            ERA5_Land_ST = read_nc_variable(ERA5_Land_ST_file , 'ERA5_Land_ST')
            ERA5_ST = read_nc_variable(ERA5_ST_file , 'ERA5_ST')
            MERRA2_ST = read_nc_variable(MERRA2_ST_file , 'MERRA2_ST') # 1980~2025
            JRA3Q_ST = read_nc_variable(JRA3Q_ST_file , 'JRA3Q_ST')
            ERA5_Land_SM = read_nc_variable(ERA5_Land_SM_file , 'ERA5_Land_SM')
            # ERA5_ST_lag3 = read_nc_variable(ERA5_ST_lag3_file , 'ERA5_ST')
            # ERA5_Land_SM_lag7 = read_nc_variable(ERA5_Land_SM_lag7_file , 'ERA5_Land_SM')
            ERA5_Land_ST_lag3 = read_nc_variable(ERA5_Land_ST_lag3_file , 'ERA5_Land_ST')
            ERA5_Land_ST_lag7 = read_nc_variable(ERA5_Land_ST_lag7_file , 'ERA5_Land_ST')
            MERRA2_ST_lag7 = read_nc_variable(MERRA2_ST_lag7_file , 'MERRA2_ST') # 1980~2025
            # Prec = read_nc_variable(Prec_file , 'Prec') 
            # NSW = read_nc_variable(NSW_file , 'NSW')
            SUBRF = read_nc_variable(SUBRF_file , 'SUBRF') 
            # SW_Down = read_nc_variable(SW_Down_file , 'SW_Down')
            # LW_Down = read_nc_variable(LW_Down_file , 'LW_Down')
            SP = read_nc_variable(SP_file , 'SP')
            T2m = read_nc_variable(T2m_file , 'T2m')
            # Td = read_nc_variable(Td_file , 'Td')
            # ET = read_nc_variable(ET_file , 'ET')
            # LH = read_nc_variable(LH_file , 'LH')
            NLW = read_nc_variable(NLW_file , 'NLW')
            # DOY_cos
            m, p = DEM.shape
            DOY_COS_2D = create_doy_cos_2d(date_str, shape=(m, p))
            # ============================ 构建特征数据集 ===============================
            feature_to_data = {
                # 动态数据
                # 'GLDAS_Noah_SKT': GLDAS_Noah_SKT.flatten(),
                # 'GLDAS_CLSM_SKT': GLDAS_CLSM_SKT.flatten(),
                # 'ERA5_Land_SKT': ERA5_Land_SKT.flatten(),
                'ERA5_SKT': ERA5_SKT.flatten(),
                'MERRA2_SKT': MERRA2_SKT.flatten(), # 1980~2025
                # 'JRA3Q_SKT': JRA3Q_SKT.flatten(),
                'GLDAS_Noah_ST': GLDAS_Noah_ST.flatten(),
                'ERA5_Land_ST': ERA5_Land_ST.flatten(),
                'ERA5_ST': ERA5_ST.flatten(),
                'MERRA2_ST': MERRA2_ST.flatten(), # 1980~2025
                'JRA3Q_ST': JRA3Q_ST.flatten(),
                'ERA5_Land_ST_lag3': ERA5_Land_ST_lag3.flatten(),
                'ERA5_Land_ST_lag7': ERA5_Land_ST_lag7.flatten(),
                'MERRA2_ST_lag7': MERRA2_ST_lag7.flatten(), # 1980~2025
                # 'ERA5_ST_lag3': ERA5_ST_lag3.flatten(),
                # 'ERA5_Land_SM_lag7': ERA5_Land_SM_lag7.flatten(),
                'ERA5_Land_SM': ERA5_Land_SM.flatten(),
                # 'Prec': Prec.flatten(),
                # 'NSW': NSW.flatten(),
                'SUBRF': SUBRF.flatten(),
                # 'SW_Down': SW_Down.flatten(),
                # 'LW_Down': LW_Down.flatten(),
                'SP': SP.flatten(),
                'T2m': T2m.flatten(),
                # 'Td': Td.flatten(),
                # 'ET': ET.flatten(),
                'NLW': NLW.flatten(),
                # 'LH': LH.flatten(),
                'DOY_cos': DOY_COS_2D.flatten(),
                # 静态数据
                'DEM': DEM.flatten(),
                # 'SLOPE': SLOPE.flatten(),
                'bd': BD.flatten(),
                'thickness': Thickness.flatten(),
                # 'ph': PH.flatten(),
                'soc': SOC.flatten(),
                'cec': CEC.flatten(),
                # 'tp': TP.flatten(),
                # 'tk': TK.flatten(),
                'tn': TN.flatten(),
                # 'cf': CF.flatten(),
                # 'btsnd': Sand.flatten(),
                # 'btslt': Silt.flatten(),
                # 'btcly': Clay.flatten(),
                # 'texcls': Texture.flatten()
            }
            # 检查所有特征是否都在 feature_to_data 中，如果缺失则抛出错误
            missing_features = [f for f in features if f not in feature_to_data]
            if missing_features:
                raise KeyError(f"Missing features in input data: {missing_features}")

            # 在调用np.column_stack之前，先检查每个数组的长度是否相同
            ref_shape = (DEM.flatten()).shape
            for name, arr in feature_to_data.items():
                if arr.shape != ref_shape:
                    raise ValueError(f"Shape mismatch for {name}: expected {ref_shape}, got {arr.shape}")
            # 按照 features 列表的顺序构建数据库
            database_ORI = [feature_to_data[feature] for feature in features if feature in feature_to_data]
            database_df = pd.DataFrame(np.column_stack(database_ORI), columns=features)
            print(database_df.columns)
            # ========================== 数据清洗：根据合理范围设置异常值为NaN ==========================
            for feature_col in database_df.columns:
                database_df.loc[abs(database_df[feature_col] > 1e9), feature_col] = np.nan
            # 将缺失值替换为中值
            for column_name in features:
                database_df[column_name] = database_df[column_name].fillna(database_df[column_name].median())
            # ============================ 将数据集分为训练集和测试集 ======================================
            # 提取特征
            X_test = database_df[features]
            # 对测试集进行标准化操作（使用训练集的统计信息）
            X_test_scaled = scaler.transform(X_test)
            # predicte test data
            predicted_data = model.predict(X_test_scaled)
            # ============================== 保存预测结果 ====================================
            # 假设原始二维数组的形状是 (m, p)，则 predicted_data 的形状是 (n,)
            m, p = DEM.shape[0], DEM.shape[1]  # 原始二维数组的形状
            ML_SM_2D = (np.reshape(predicted_data , (m, p))) # 将 predicted_data 转换为形状 (m, p)
            ML_SM_2D[ML_SM_2D < 0] = 0
            ML_SM_2D = ML_SM_2D*mask #只保留中国大陆区域
            #保存数据
            date = pd.to_datetime(date_str)
            create_nc_file(date, lat, lon, ML_SM_2D, GRIDDATA_FILEPATH, MODEL_NAME, SOIL_DEPTH)
            end_merge = time.time()
            merge_time = round(end_merge - start_merge , 2)
            print(f"Success merged {date}.{MODEL_NAME}_ST000010.nc----Time: {merge_time}Seconds")
            pass
# =============================================================================
# 
import calendar
import argparse
from concurrent.futures import ProcessPoolExecutor
def merge_data_task(args):
    Grid_data_for_year(*args)
    pass

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--end_year", type=int, required=True)
    parser.add_argument("--soil_depth", type=str, default="0cm")
    parser.add_argument("--model_name", type=str, default="CB")
    parser.add_argument("--start_month", type=int, default=1)
    parser.add_argument("--end_month", type=int, default=12)
    parser.add_argument("--dst_resolution", type=str, default="1km")
    args = parser.parse_args()

    SOIL_DEPTH = args.soil_depth
    START_YEAR = args.start_year
    END_YEAR = args.end_year
    START_MONTH = args.start_month
    END_MONTH = args.end_month
    MODEL_NAME = args.model_name
    DSTresolution = args.dst_resolution

    # # # 手动设置参数
    # SOIL_DEPTH = '10cm'
    # MaxWorks = 5
    # START_YEAR = 2020
    # END_YEAR = 2024
    # START_MONTH = 1
    # END_MONTH = 12
    # MODEL_NAME = "CB"
    # merge_args = []

    # 固定参数
    DSTresolution = "1km"  
    OUTPUT_FOLDER = "default"
    feature_selection_strategy = "RFE"
    CAL_IMP_METHOD = "shap"
    print("==================================================================")
    print(f"Processing data for {MODEL_NAME} with soil depth {SOIL_DEPTH}...")
    print(f"Starting from {START_YEAR} to {END_YEAR}...")
    print(f"Processing from {START_MONTH} to {END_MONTH}...")
    print("==================================================================")
    for year in range(START_YEAR, END_YEAR+1):
        args = (year, START_MONTH, END_MONTH, SOIL_DEPTH, MODEL_NAME, DSTresolution)
        # merge_args.append(args)
        Grid_data_for_year(*args)
    # # 使用 tqdm 包装 executor.map()
    # from tqdm import tqdm
    # with ProcessPoolExecutor(max_workers=MaxWorks) as executor:
    #     # executor.map 返回一个 iterator，tqdm 可以跟踪进度
    #     list(tqdm(executor.map(merge_data_task, merge_args), total=len(merge_args), desc="Merging Data", colour="blue"))
# =============================================================================

end = time.time()
print(f"Elapse Time: {end - start}Seconds")