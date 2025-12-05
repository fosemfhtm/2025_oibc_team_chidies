# ==========================================
# 라이브러리 임포트
# ==========================================
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree
import os

# ==========================================
# 설정 (Configuration)
# ==========================================
K_NEIGHBORS = 100   # 잔차 보간에 사용할 최근접 이웃 수
IDW_POWER = 2.0     # IDW 가중치 지수 (p)

TRAIN_TREND_FILE = '5_1_train_with_trend.parquet'
TEST_TREND_FILE = '5_1_test_with_trend.parquet'
SUBMISSION_FILE = 'submission_sample.csv'

# ==========================================
# 데이터 로드
# ==========================================
print("=" * 70)
print("5-2단계: 잔차에 대한 IDW 보간 및 최종 예측 (최적화 버전)")
print("=" * 70)

try:
    train_df = pd.read_parquet(TRAIN_TREND_FILE)
    test_df = pd.read_parquet(TEST_TREND_FILE)
    submission = pd.read_csv(SUBMISSION_FILE)
    print(f"✅ 데이터 로딩 완료")
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다.")
    exit()

# ==========================================
# [핵심] 고속 IDW 보간 함수 (벡터화 적용)
# ==========================================
def fast_idw_interpolation(train_data, test_data, k=100, p=2.0):
    """
    Numpy Broadcasting을 사용하여 루프 없이 고속으로 잔차를 보간합니다.
    """
    print("\n[최적화된 IDW 예측 시작]")
    
    # 1. 데이터 준비 (Pivot: 행=시간, 열=발전소ID)
    print(" 1. 학습 데이터 Pivot 변환 중...")
    train_pivot = train_data.pivot(index='time', columns='pv_id', values='residual')
    
    # 2. 위치 정보 추출 (중복 제거)
    train_locs = train_data[['pv_id', 'coord1', 'coord2']].drop_duplicates().set_index('pv_id')
    test_locs = test_data[['pv_id', 'coord1', 'coord2']].drop_duplicates().set_index('pv_id')
    
    # train_pivot의 컬럼 순서와 train_locs의 순서를 일치시킴
    train_locs = train_locs.loc[train_pivot.columns]
    
    # 3. 공간 검색 (KDTree) - *발전소 별로 딱 한 번만 수행*
    print(" 2. 공간 이웃 검색 (KDTree)...")
    tree = cKDTree(train_locs[['coord1', 'coord2']].values)
    
    # 모든 테스트 발전소에 대해 K개 이웃 찾기
    dists, indices = tree.query(test_locs[['coord1', 'coord2']].values, k=k)
    
    # 4. 가중치 미리 계산 (행렬 연산)
    print(" 3. 가중치 행렬 계산...")
    weights = 1.0 / (dists ** p + 1e-6)
    
    # 5. 시간대별 잔차 매핑 및 예측
    print(" 4. 잔차 보간 계산 (행렬 연산)...")
    
    test_data_sorted = test_data.sort_values(['time', 'pv_id'])
    unique_times = test_data_sorted['time'].unique()
    
    # Train 잔차 매트릭스에서 Test에 필요한 시간대만 가져옴 (Reindex)
    current_residuals_matrix = train_pivot.reindex(unique_times).values 
    
    # neighbor_residuals Shape: (Time, Test_Stations, K)
    neighbor_residuals = current_residuals_matrix[:, indices] 
    
    # NaN 마스킹
    mask = ~np.isnan(neighbor_residuals)
    neighbor_residuals_filled = np.nan_to_num(neighbor_residuals, nan=0.0)
    
    # 가중치 적용 (Broadcasting)
    weights_expanded = weights[np.newaxis, :, :]
    valid_weights = weights_expanded * mask
    
    # 가중 평균 계산
    weighted_sum = np.sum(neighbor_residuals_filled * valid_weights, axis=2)
    sum_of_weights = np.sum(valid_weights, axis=2)
    
    final_residual_matrix = np.divide(
        weighted_sum, 
        sum_of_weights, 
        out=np.zeros_like(weighted_sum), 
        where=sum_of_weights != 0
    )
    
    # 6. 결과 매핑
    result_df = pd.DataFrame(
        final_residual_matrix, 
        index=unique_times, 
        columns=test_locs.index
    ).stack().reset_index()
    
    result_df.columns = ['time', 'pv_id', 'interpolated_residual']
    
    merged = pd.merge(test_data, result_df, on=['time', 'pv_id'], how='left')
    merged['interpolated_residual'] = merged['interpolated_residual'].fillna(0)
    
    return merged['interpolated_residual'].values

# ==========================================
# 실행
# ==========================================
# 1. 최종 예측 (Test 셋)
test_residuals = fast_idw_interpolation(train_df, test_df, k=K_NEIGHBORS, p=IDW_POWER)
final_test_predictions = test_df['trend_nins'] + test_residuals
final_test_predictions = np.clip(final_test_predictions, 0, None)

print("\n✅ 최종 예측 완료")

# 2. 검증 (Validation)
print("\n" + "=" * 70)
print("검증 수행 (Hold-out)...")
train_ids, val_ids = train_test_split(train_df['pv_id'].unique(), test_size=0.2, random_state=42)
train_main_df = train_df[train_df['pv_id'].isin(train_ids)]
val_df = train_df[train_df['pv_id'].isin(val_ids)]

val_interpolated = fast_idw_interpolation(train_main_df, val_df, k=K_NEIGHBORS, p=IDW_POWER)
val_predictions = val_df['trend_nins'] + val_interpolated
val_predictions = np.clip(val_predictions, 0, None)

mae_trend = mean_absolute_error(val_df['nins'], val_df['trend_nins'])
mae_final = mean_absolute_error(val_df['nins'], val_predictions)

print("\n[검증 결과]")
print(f" - 추세 MAE: {mae_trend:.4f}")
print(f" - 최종 MAE: {mae_final:.4f}")
print(f" - 개선폭: {mae_trend - mae_final:.4f}")

# ==========================================
# 제출 파일 생성
# ==========================================
print("\n" + "=" * 70)
print("제출 파일 생성...")
print("=" * 70)

submission['nins'] = final_test_predictions
submission_filename = '5_2_regression_IDW_submission.csv'  # <--- 요청하신 대로 파일명 유지
submission.to_csv(submission_filename, index=False)

print(f"✅ 제출 파일 저장 완료: '{submission_filename}'")
print(f" - 최종 예측 결과 평균: {submission['nins'].mean():.4f}")
print("=" * 70)