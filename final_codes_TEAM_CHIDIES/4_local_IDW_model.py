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
# 예측에 사용할 가장 가까운 이웃 발전소의 수
K_NEIGHBORS = 50
# IDW 가중치 계산 시 거리의 역수에 적용할 지수 (p). 클수록 가까운 이웃의 영향력이 커짐.
IDW_POWER = 2.0
# 검증 시 사용할 샘플 크기
VALIDATION_SAMPLE_SIZE = 10000

# ==========================================
# 데이터 로드
# ==========================================
print("=" * 50)
print("데이터 로딩 시작...")
print("=" * 50)

try:
    train = pd.read_csv('train.csv', usecols=['time', 'pv_id', 'coord1', 'coord2', 'nins'])
    test = pd.read_csv('test.csv', usecols=['time', 'pv_id', 'coord1', 'coord2'])
    submission = pd.read_csv('submission_sample.csv')
    
    train['time'] = pd.to_datetime(train['time'])
    test['time'] = pd.to_datetime(test['time'])
    print("✅ 데이터 로딩 및 기본 변환 완료")
except FileNotFoundError as e:
    print(f"❌ 파일 로딩 오류: {e}.")
    exit()

# ==========================================
# IDW 모델 준비
# ==========================================
print("\n" + "=" * 50)
print("IDW 모델 준비...")
print("=" * 50)

# 1. 발전소 위치 정보 추출
train_locations = train[['pv_id', 'coord1', 'coord2']].drop_duplicates().set_index('pv_id')
test_locations = test[['pv_id', 'coord1', 'coord2']].drop_duplicates().set_index('pv_id')
print(f"  - 학습 발전소: {len(train_locations)}개, 테스트 발전소: {len(test_locations)}개")

# 2. 공간 검색을 위한 KD-Tree 생성
print("  - 학습 발전소 위치로 KD-Tree 생성 중...")
kdtree = cKDTree(train_locations[['coord1', 'coord2']].values)

# 3. 각 테스트 발전소의 최근접 이웃 탐색
print(f"  - 각 테스트 발전소의 최근접 이웃 {K_NEIGHBORS}개 탐색 중...")
distances, indices = kdtree.query(test_locations[['coord1', 'coord2']].values, k=K_NEIGHBORS)

# 4. 이웃 정보 딕셔너리 생성
neighbor_map = {
    test_pv_id: {'ids': train_locations.index[indices[i]], 'dists': distances[i]}
    for i, test_pv_id in enumerate(test_locations.index)
}
print("✅ IDW 모델 준비 완료")

# ==========================================
# 예측 함수 정의
# ==========================================
def idw_predict(target_time, target_pv_id, train_pivot, neighbor_info):
    """특정 시간, 특정 발전소에 대해 IDW 예측값을 계산"""
    neighbors = neighbor_info.get(target_pv_id)
    if not neighbors:
        return 0.0
        
    try:
        neighbor_nins = train_pivot.loc[target_time, neighbors['ids']].values
    except KeyError:
        return 0.0

    valid_mask = ~np.isnan(neighbor_nins)
    if not np.any(valid_mask):
        return 0.0
        
    valid_nins = neighbor_nins[valid_mask]
    valid_dists = neighbors['dists'][valid_mask]
    
    # IDW 가중치 계산: weights = 1 / (distance^p)
    weights = 1.0 / (valid_dists**IDW_POWER + 1e-6)
    
    # 가중 평균 계산
    return np.average(valid_nins, weights=weights)

# ==========================================
# 최종 예측
# ==========================================
print("\n" + "=" * 50)
print("최종 예측 수행...")
print("=" * 50)

# 빠른 조회를 위해 train 데이터를 pivot
train_pivot = train.pivot(index='time', columns='pv_id', values='nins')

# apply 함수를 사용하여 예측 수행
test['nins'] = test.apply(
    lambda row: idw_predict(row['time'], row['pv_id'], train_pivot, neighbor_map),
    axis=1
)
test['nins'] = test['nins'].clip(lower=0)
print("✅ 최종 예측 완료")

# ==========================================
# 검증 (Validation)
# ==========================================
print("\n" + "=" * 50)
print("검증 수행...")
print("=" * 50)

# 1. 검증 데이터 분리
train_main_ids, val_ids = train_test_split(train_locations.index, test_size=0.2, random_state=42)
train_main_loc = train_locations.loc[train_main_ids]
val_loc = train_locations.loc[val_ids]

# 2. 검증용 KD-Tree 및 이웃 탐색
val_kdtree = cKDTree(train_main_loc[['coord1', 'coord2']].values)
val_dists, val_indices = val_kdtree.query(val_loc[['coord1', 'coord2']].values, k=K_NEIGHBORS)
val_neighbor_map = {
    val_pv_id: {'ids': train_main_loc.index[val_indices[i]], 'dists': val_dists[i]}
    for i, val_pv_id in enumerate(val_loc.index)
}

# 3. 검증 데이터 예측
val_data = train[train['pv_id'].isin(val_ids)].sample(n=VALIDATION_SAMPLE_SIZE, random_state=42)
val_train_pivot = train[train['pv_id'].isin(train_main_ids)].pivot(index='time', columns='pv_id', values='nins')

val_data['pred_nins'] = val_data.apply(
    lambda row: idw_predict(row['time'], row['pv_id'], val_train_pivot, val_neighbor_map),
    axis=1
).clip(lower=0)

# 4. MAE 계산
mae = mean_absolute_error(val_data['nins'], val_data['pred_nins'])
print(f"  - 검증 MAE (k={K_NEIGHBORS}, p={IDW_POWER}): {mae:.4f}")
print("✅ 검증 완료")

# ==========================================
# 제출 파일 생성
# ==========================================
print("\n" + "=" * 50)
print("제출 파일 생성...")
print("=" * 50)

submission['nins'] = test['nins'].values
submission_filename = '4_local_IDW_submission.csv'
submission.to_csv(submission_filename, index=False)

print(f"✅ 제출 파일 저장 완료: '{submission_filename}'")
print(f"  - 예측 결과 평균: {submission['nins'].mean():.4f}")
print("=" * 50)