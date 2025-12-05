# ==========================================
# 라이브러리 임포트
# ==========================================
import pandas as pd
import numpy as np
import os
import gc
from lightgbm import LGBMRegressor, log_evaluation
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ==========================================
# 실행 환경 설정
# ==========================================
MODE = 'final'
MODEL_TYPE = 'lightgbm'

# ==========================================
# 모델 하이퍼파라미터
# ==========================================
LGBM_PARAMS = {
    'n_estimators': 5000, 'learning_rate': 0.01, 'max_depth': 10,
    'num_leaves': 50, 'min_child_samples': 20, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'random_state': 42, 'n_jobs': -1
}
XGB_PARAMS = {
    'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 10,
    'min_child_weight': 20, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'random_state': 42,
    'n_jobs': -1, 'eval_metric': 'mae'
}

# ==========================================
# 파일 경로 설정
# ==========================================
PROCESSED_TRAIN_FILE = '1_processed_train_interpolate.parquet'
PROCESSED_TEST_FILE = '1_processed_test_interpolate.parquet'
SUBMISSION_FILE = 'submission_sample.csv'

# ==========================================
# 데이터 로드
# ==========================================
print("=" * 50)
print("1단계 모델: 선형 보간")
print("=" * 50)
print("데이터 로딩 시작...")

try:
    # 메모리 최적화를 위해 필요한 컬럼만 로드하거나 로드 후 즉시 변환
    # 여기서는 전체 로드 후 float32로 변환 및 불필요 컬럼 제거 방식 사용
    print("  - Train 데이터 로드 중...")
    train = pd.read_parquet(PROCESSED_TRAIN_FILE)
    
    # float64 -> float32 변환 (Train)
    float_cols_train = train.select_dtypes(include=['float64']).columns
    train[float_cols_train] = train[float_cols_train].astype('float32')
    
    print("  - Test 데이터 로드 중...")
    test = pd.read_parquet(PROCESSED_TEST_FILE)
    # float64 -> float32 변환 (Test)
    float_cols_test = test.select_dtypes(include=['float64']).columns
    test[float_cols_test] = test[float_cols_test].astype('float32')
    
    submission = pd.read_csv(SUBMISSION_FILE)
    print("✅ 데이터 로딩 및 최적화 완료")
except FileNotFoundError:
    print(f"❌ 전처리된 파일을 찾을 수 없습니다. `1_1_preprocess_data.py`를 먼저 실행하세요.")
    exit()

# 학습에 사용할 특성 리스트 정의
features = [col for col in train.columns if col not in ['time', 'pv_id', 'type', 'energy', 'nins']]
print(f"  - 학습에 사용할 특성 개수: {len(features)}")

# 메모리 절약을 위해 불필요한 컬럼 제거 (time, type, energy 등)
# 단, pv_id는 검증 모드 분할에 필요하므로 유지, nins는 타겟이므로 유지
cols_to_drop = ['time', 'type', 'energy']
train.drop(columns=[c for c in cols_to_drop if c in train.columns], inplace=True)
test.drop(columns=[c for c in cols_to_drop if c in test.columns], inplace=True)
gc.collect()

# ==========================================
# 모델 학습 및 예측
# ==========================================
# 모델 타입에 따라 파라미터 및 모델 선택
if MODEL_TYPE == 'lightgbm':
    model = LGBMRegressor(**LGBM_PARAMS)
    print("  - 모델 타입: LightGBM")
elif MODEL_TYPE == 'xgboost':
    model = XGBRegressor(**XGB_PARAMS)
    print("  - 모델 타입: XGBoost")
else:
    raise ValueError(f"알 수 없는 모델 타입: {MODEL_TYPE}")

if MODE == 'validation':
    # --- 검증 모드 ---
    print("\n" + "=" * 50)
    print("검증 모드 실행...")
    print("=" * 50)

    # 1. 발전소 ID 기준 8:2 분할
    unique_pv_ids = train['pv_id'].unique()
    train_pv_ids, val_pv_ids = train_test_split(unique_pv_ids, test_size=0.2, random_state=42)

    train_data = train[train['pv_id'].isin(train_pv_ids)]
    val_data = train[train['pv_id'].isin(val_pv_ids)]

    X_train = train_data[features]
    y_train = train_data['nins']
    X_val = val_data[features]
    y_val = val_data['nins']
    
    print(f"  - 학습 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")

    # 2. 모델 학습 및 검증
    print("  - 모델 학습 중...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae', callbacks=[log_evaluation(500)])

    val_pred = model.predict(X_val)
    
    # 후처리: 물리적 제약 조건 적용
    val_pred[val_pred < 0] = 0
    night_hours = [0, 1, 2, 3, 4, 5, 20, 21, 22, 23]
    night_mask = val_data['hour'].isin(night_hours)
    val_pred[night_mask] = 0
    
    mae = mean_absolute_error(y_val, val_pred)
    print(f"\n[검증 성능]")
    print(f"  - MAE (Mean Absolute Error): {mae:.4f}")
    print("✅ 검증 완료")

elif MODE == 'final':
    # --- 최종 모드 ---
    print("\n" + "=" * 50)
    print("최종 모드 실행...")
    print("=" * 50)

    # 1. 전체 학습 데이터로 모델 학습
    # 메모리 효율을 위해 데이터프레임에서 직접 numpy array로 변환하지 않고 LightGBM Dataset 사용 고려
    # 하지만 sklearn API에서는 X, y를 넘겨야 함.
    
    # 불필요한 pv_id 제거하여 메모리 확보
    if 'pv_id' in train.columns:
        train.drop(columns=['pv_id'], inplace=True)
    gc.collect()

    X_train_full = train[features]
    y_train_full = train['nins']
    
    # train 데이터프레임 삭제로 메모리 확보
    del train
    gc.collect()
    
    print(f"  - 전체 학습 데이터로 모델 학습 중... (데이터: {X_train_full.shape})")
    model.fit(X_train_full, y_train_full, callbacks=[log_evaluation(500)])

    # 2. 테스트 데이터 예측
    print("  - 테스트 데이터로 최종 예측 수행 중...")
    test_pred = model.predict(test[features])
    
    # 후처리
    test_pred[test_pred < 0] = 0
    night_hours = [0, 1, 2, 3, 4, 5, 20, 21, 22, 23]
    night_mask = test['hour'].isin(night_hours)
    test_pred[night_mask] = 0

    # 3. 제출 파일 생성
    submission['nins'] = test_pred
    submission_filename = '1_interpolate_submission.csv'
    submission.to_csv(submission_filename, index=False)

    print(f"\n✅ 최종 제출 파일 저장 완료: '{submission_filename}'")
    print("=" * 50)
