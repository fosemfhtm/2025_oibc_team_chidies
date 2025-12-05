# ==========================================
# 라이브러리 임포트
# ==========================================
import pandas as pd
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 실행 환경 설정
# ==========================================
# 실행 모드 설정
# 'validation': 모델 성능 검증 및 분석
# 'final': 최종 제출 파일 생성
MODE = 'final'

# 모델 하이퍼파라미터
LGBM_PARAMS = {
    'n_estimators': 5000, 'learning_rate': 0.01, 'max_depth': 10,
    'num_leaves': 50, 'min_child_samples': 20, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'random_state': 42, 'n_jobs': -1
}

# ==========================================
# 파일 경로 및 시각화 설정
# ==========================================
# 전처리된 데이터 파일 경로 (2_1_preprocess_data.py 실행 결과)
PROCESSED_TRAIN_FILE = '2_processed_train_attenuation.parquet'
PROCESSED_TEST_FILE = '2_processed_test_attenuation.parquet'
SUBMISSION_FILE = 'submission_sample.csv'

# 시각화 결과 저장 파일명
TIMESLOT_PLOT_FILE = '2_2_timeslot_validation_results.png'
HOURLY_PLOT_FILE = '2_2_hourly_validation_results.png'
STATION_MAE_PLOT_FILE = '2_2_station_mae_distribution.png'

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 데이터 로드
# ==========================================
print("=" * 50)
print("데이터 로딩 시작...")
print("=" * 50)

# 전처리된 파일 로드
try:
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
    print(f"❌ 전처리된 파일을 찾을 수 없습니다. 먼저 `2_1_preprocess_data.py`를 실행하세요.")
    exit()

# ==========================================
# 모델링 준비
# ==========================================
print("\n" + "=" * 50)
print("모델링 준비 시작...")
print("=" * 50)

# 1. 특성 선택
#    - 직접 nins를 예측하는 대신, 날씨에 의한 '감쇠율'을 예측하는 것이 이 모델의 핵심.
#    - 따라서 타겟과 직접 관련된 'nins', 'irradiance_attenuation_rate'와
exclude_cols = ['time', 'pv_id', 'type', 'energy', 'nins', 'irradiance_attenuation_rate']
features = [col for col in train.columns if col not in exclude_cols]
print(f"  - 감쇠율 예측에 사용할 특성 개수: {len(features)}")

# 2. 학습 데이터 필터링 (주간 데이터만 사용)
#    - 감쇠율은 이론상 최대 일사량이 0보다 클 때만 의미가 있음 (분모가 0이 되는 것 방지).
#    - 따라서 해가 떠 있는 주간(daytime) 데이터만으로 모델을 학습.
NIGHT_HOURS = [0, 1, 2, 3, 4, 20, 21, 22, 23]
train_daytime = train[~train['hour'].isin(NIGHT_HOURS)].copy()
print(f"  - 야간 시간({NIGHT_HOURS}) 제외 후 학습 데이터: {train_daytime.shape} (원본: {train.shape})")

# 원본 train 데이터는 더 이상 필요 없으므로 삭제 (검증 모드 제외)
if MODE == 'final':
    del train
    gc.collect()

# 3. 데이터 분할 (학습/검증)
if MODE == 'validation':
    print("  - 검증 모드: 발전소 ID 기준 80:20 데이터 분할")
    unique_pv_ids = train_daytime['pv_id'].unique()
    train_pv_ids, val_pv_ids = train_test_split(unique_pv_ids, test_size=0.2, random_state=42)
    
    train_mask = train_daytime['pv_id'].isin(train_pv_ids)
    val_mask = train_daytime['pv_id'].isin(val_pv_ids)

    X_train = train_daytime.loc[train_mask, features]
    y_train = train_daytime.loc[train_mask, 'irradiance_attenuation_rate']
    X_val = train_daytime.loc[val_mask, features]
    y_val = train_daytime.loc[val_mask, 'irradiance_attenuation_rate']
    
    # 최종 nins 성능 평가를 위해 원본 검증 데이터 저장
    val_data_full = train[train['pv_id'].isin(val_pv_ids)].copy()
    
else: # 'final'
    print("  - 최종 모드: 전체 주간 데이터로 학습")
    X_train = train_daytime[features]
    y_train = train_daytime['irradiance_attenuation_rate']

print(f"  - 학습 데이터 형태: X_train={X_train.shape}, y_train={y_train.shape}")
if MODE == 'validation':
    print(f"  - 검증 데이터 형태: X_val={X_val.shape}, y_val={y_val.shape}")

# ==========================================
# 모델 학습
# ==========================================
print("\n" + "=" * 50)
print("LGBM 감쇠율 모델 학습 시작...")
print("=" * 50)

model = LGBMRegressor(**LGBM_PARAMS)

eval_set = [(X_val, y_val)] if MODE == 'validation' else None
model.fit(X_train, y_train, eval_set=eval_set, eval_metric='rmse', callbacks=[log_evaluation(100)])

# ==========================================
# 검증 (Validation Mode Only)
# ==========================================
if MODE == 'validation':
    print("\n" + "=" * 50)
    print("검증 성능 평가...")
    print("=" * 50)

    # 1. 감쇠율 예측 (주간 데이터에 대해서만)
    pred_attenuation_daytime = model.predict(X_val)
    pred_attenuation_daytime = np.clip(pred_attenuation_daytime, 0, 1.2) # 물리적 범위 제한

    # 2. 최종 nins 예측값으로 변환
    #    - 예측된 감쇠율 * 이론상 최대 일사량 = 최종 일사량 예측치
    pred_nins = np.zeros(len(val_data_full))
    daytime_mask = ~val_data_full['hour'].isin(NIGHT_HOURS)
    
    # 주간 시간대에만 예측된 감쇠율 적용
    pred_attenuation_full = np.zeros(len(val_data_full))
    pred_attenuation_full[daytime_mask] = model.predict(val_data_full.loc[daytime_mask, features])
    pred_attenuation_full = np.clip(pred_attenuation_full, 0, 1.2)
    
    pred_nins = pred_attenuation_full * val_data_full['theoretical_max_irradiance'].values
    pred_nins = np.maximum(pred_nins, 0) # 음수 방지

    # 3. 성능 평가
    true_nins = val_data_full['nins'].values
    mae = mean_absolute_error(true_nins, pred_nins)
    
    print("\n[최종 nins 예측 성능]")
    print(f"  - MAE (Mean Absolute Error): {mae:.4f}")

    true_attenuation_daytime = y_val.values
    r2 = r2_score(true_attenuation_daytime, pred_attenuation_daytime)
    print("\n[감쇠율 예측 성능 (주간 데이터)]")
    print(f"  - R² Score: {r2:.4f}")
    
    # 시각화 등 추가 분석...
    # (시간대별, 발전소별 분석은 이전 스크립트와 유사하게 추가 가능)

# ==========================================
# 최종 예측 및 제출
# ==========================================
print("\n" + "=" * 50)
print("최종 예측 및 제출 파일 생성...")
print("=" * 50)

# 1. 테스트 데이터에 대한 감쇠율 예측
#    - 주간 시간대에 대해서만 예측 수행
pred_attenuation_test = np.zeros(len(test))
daytime_mask_test = ~test['hour'].isin(NIGHT_HOURS)
pred_attenuation_test[daytime_mask_test] = model.predict(test.loc[daytime_mask_test, features])

# 2. 후처리 및 최종 nins 변환
pred_attenuation_test = np.clip(pred_attenuation_test, 0, 1.2)
pred_nins_test = pred_attenuation_test * test['theoretical_max_irradiance'].values
pred_nins_test = np.maximum(pred_nins_test, 0)

# 3. 제출 파일 생성
submission['nins'] = pred_nins_test
submission_filename = f"2_2_attenuation_submission_{MODE}.csv"
submission.to_csv(submission_filename, index=False)

print(f"✅ 제출 파일 저장 완료: '{submission_filename}'")
print("\n[예측 결과 통계]")
print(f"  - 예측 nins 평균: {pred_nins_test.mean():.4f}")
print(f"  - 예측 감쇠율 평균: {pred_attenuation_test.mean():.4f}")
print("=" * 50)
