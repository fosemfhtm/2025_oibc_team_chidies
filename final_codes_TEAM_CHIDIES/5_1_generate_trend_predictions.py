# ==========================================
# 라이브러리 임포트
# ==========================================
import pandas as pd
import numpy as np
import os
import joblib
from lightgbm import LGBMRegressor, log_evaluation

# ==========================================
# 설정 (Configuration)
# ==========================================
# 이 스크립트는 2단계 모델(잔차 회귀)의 첫 번째 단계입니다.
# 목적: 공간 정보(좌표)를 제외한 기상 및 시간 특성만을 사용하여
#      일반적인 '일사량 추세(trend)'를 예측하는 모델을 학습하고, 그 예측 결과를 저장합니다.
#      이렇게 생성된 '추세'는 2단계에서 '잔차(residual)'를 계산하는 데 사용됩니다.

# 모델 하이퍼파라미터
LGBM_PARAMS = {
    'n_estimators': 5000, 'learning_rate': 0.01, 'max_depth': 10,
    'num_leaves': 50, 'min_child_samples': 20, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'random_state': 42, 'n_jobs': -1, 'verbose': -1
}

# 파일 경로
PROCESSED_TRAIN_FILE = '2_processed_train_attenuation.parquet'
PROCESSED_TEST_FILE = '2_processed_test_attenuation.parquet'
MODEL_FILE = '5_1_trend_model.pkl'
OUTPUT_TRAIN_TREND = '5_1_train_with_trend.parquet'
OUTPUT_TEST_TREND = '5_1_test_with_trend.parquet'

# 야간 시간 (학습에서 제외)
NIGHT_HOURS = [0, 1, 2, 3, 4, 20, 21, 22, 23]

# ==========================================
# 데이터 로드
# ==========================================
print("=" * 70)
print("5-1단계: 일사량 추세 예측 모델")
print("=" * 70)
print("데이터 로딩 중...")

try:
    train = pd.read_parquet(PROCESSED_TRAIN_FILE)
    test = pd.read_parquet(PROCESSED_TEST_FILE)
    print(f"✅ 데이터 로딩 완료 (Train: {train.shape}, Test: {test.shape})")
except FileNotFoundError:
    print(f"❌ 전처리된 파일을 찾을 수 없습니다. `2_1_preprocess_data.py`를 먼저 실행하세요.")
    exit()

# ==========================================
# 모델 학습 또는 로드
# ==========================================
print("\n" + "=" * 70)
print("추세 예측 모델 학습 또는 로드")
print("=" * 70)

# 학습에 사용할 특성 선택 (공간 정보인 coord1, coord2 제외)
exclude_cols = ['time', 'pv_id', 'type', 'energy', 'nins', 
                'irradiance_attenuation_rate', 'coord1', 'coord2']
features = [col for col in train.columns if col not in exclude_cols]
print(f"  - 학습에 사용할 특성 개수: {len(features)}")

if os.path.exists(MODEL_FILE):
    print(f"  - 기존 모델 로딩: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)
else:
    print("  - 새로운 모델 학습 중...")
    # 주간 데이터만 학습에 사용
    daytime_mask = ~train['hour'].isin(NIGHT_HOURS)
    train_daytime = train.loc[daytime_mask]
    
    X_train = train_daytime[features]
    # 감쇠율을 타겟으로 학습
    y_train = train_daytime['irradiance_attenuation_rate']
    
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train, callbacks=[log_evaluation(500)])
    
    joblib.dump(model, MODEL_FILE)
    print(f"  - 모델 학습 완료 및 저장: {MODEL_FILE}")

# ==========================================
# 추세 예측 및 잔차 계산
# ==========================================
def predict_trend_and_get_residuals(df, features, model, is_train=True):
    """데이터프레임에 대한 추세 예측 및 잔차 계산"""
    # 1. 감쇠율 예측
    pred_attenuation = np.zeros(len(df))
    daytime_mask = ~df['hour'].isin(NIGHT_HOURS)
    pred_attenuation[daytime_mask] = model.predict(df.loc[daytime_mask, features])
    pred_attenuation = np.clip(pred_attenuation, 0, 1.2)

    # 2. 일사량 추세(trend_nins) 계산
    df['trend_nins'] = pred_attenuation * df['theoretical_max_irradiance']
    df['trend_nins'] = df['trend_nins'].clip(lower=0)
    
    # 3. 잔차(residual) 계산 (학습 데이터에만 해당)
    if is_train:
        df['residual'] = df['nins'] - df['trend_nins']
    
    return df

# --- 학습 데이터에 대한 추세 예측 ---
print("\n" + "=" * 70)
print("학습 데이터에 대한 추세 예측 및 잔차 계산")
print("=" * 70)
train = predict_trend_and_get_residuals(train, features, model, is_train=True)

# 다음 단계를 위해 필요한 컬럼만 저장
train_output_cols = ['time', 'pv_id', 'coord1', 'coord2', 'hour', 'nins', 'trend_nins', 'residual']
train[train_output_cols].to_parquet(OUTPUT_TRAIN_TREND, index=False)
print(f"✅ 학습 데이터 예측 결과 저장: {OUTPUT_TRAIN_TREND}")
print(f"  - 추세 예측 범위: [{train['trend_nins'].min():.2f}, {train['trend_nins'].max():.2f}]")
print(f"  - 잔차 범위: [{train['residual'].min():.2f}, {train['residual'].max():.2f}]")

# --- 테스트 데이터에 대한 추세 예측 ---
print("\n" + "=" * 70)
print("테스트 데이터에 대한 추세 예측")
print("=" * 70)
test = predict_trend_and_get_residuals(test, features, model, is_train=False)

# 다음 단계를 위해 필요한 컬럼만 저장
test_output_cols = ['time', 'pv_id', 'coord1', 'coord2', 'hour', 'trend_nins']
test[test_output_cols].to_parquet(OUTPUT_TEST_TREND, index=False)
print(f"✅ 테스트 데이터 예측 결과 저장: {OUTPUT_TEST_TREND}")
print(f"  - 추세 예측 범위: [{test['trend_nins'].min():.2f}, {test['trend_nins'].max():.2f}]")

print("\n" + "=" * 70)
print("✅ 5-1단계 완료!")
print("💡 다음 단계: `5_2_regression_IDW_model.py`를 실행하여 잔차를 보간하고 최종 예측을 수행하세요.")
print("=" * 70)
