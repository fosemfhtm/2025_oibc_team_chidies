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
K_NEIGHBORS = 5
# 검증 시 사용할 샘플 크기 (속도 조절용)
VALIDATION_SAMPLE_SIZE = 10000

# ==========================================
# 데이터 로드
# ==========================================
print("=" * 50)
print("데이터 로딩 시작...")
print("=" * 50)

try:
    # 필요한 컬럼만 효율적으로 로드
    train = pd.read_csv('train.csv', usecols=['time', 'pv_id', 'coord1', 'coord2', 'nins'])
    test = pd.read_csv('test.csv', usecols=['time', 'pv_id', 'coord1', 'coord2'])
    submission = pd.read_csv('submission_sample.csv')
    
    # 시간 데이터타입 변환
    train['time'] = pd.to_datetime(train['time'])
    test['time'] = pd.to_datetime(test['time'])
    print("✅ 데이터 로딩 및 기본 변환 완료")
except FileNotFoundError as e:
    print(f"❌ 파일 로딩 오류: {e}. 스크립트 실행 위치를 확인하세요.")
    exit()

# ==========================================
# 모델링 준비 (공간 데이터 처리)
# ==========================================
print("\n" + "=" * 50)
print("최근접 이웃 모델링 준비...")
print("=" * 50)

# 1. 발전소 위치 정보 추출
#    - 각 발전소의 고유한 위치(coord1, coord2) 정보를 추출
train_locations = train[['pv_id', 'coord1', 'coord2']].drop_duplicates().set_index('pv_id')
test_locations = test[['pv_id', 'coord1', 'coord2']].drop_duplicates().set_index('pv_id')
print(f"  - 학습 발전소: {len(train_locations)}개, 테스트 발전소: {len(test_locations)}개")

# 2. 공간 검색을 위한 KD-Tree 생성
#    - 학습 발전소들의 위치 정보를 기반으로 KD-Tree를 생성.
#    - KD-Tree는 특정 위치에서 가장 가까운 점들을 효율적으로 찾는 데 사용되는 자료구조.
print("  - 학습 발전소 위치로 KD-Tree 생성 중...")
kdtree = cKDTree(train_locations[['coord1', 'coord2']].values)

# 3. 각 테스트 발전소의 최근접 이웃 탐색
#    - 각 테스트 발전소 위치에서 가장 가까운 K개의 학습 발전소를 탐색.
print(f"  - 각 테스트 발전소에 대한 {K_NEIGHBORS}개의 최근접 이웃 탐색 중...")
distances, indices = kdtree.query(test_locations[['coord1', 'coord2']].values, k=K_NEIGHBORS)

# 4. 이웃 정보 저장
#    - 나중에 예측 시 빠르게 사용하기 위해 각 테스트 발전소의 이웃 ID와 거리를 딕셔너리에 저장.
neighbor_map = {}
for i, test_pv_id in enumerate(test_locations.index):
    neighbor_ids = train_locations.index[indices[i]]
    neighbor_map[test_pv_id] = {
        'ids': neighbor_ids,
        'dists': distances[i]
    }
print("✅ 모델링 준비 완료")

# ==========================================
# 예측 (가중 평균 기반)
# ==========================================
print("\n" + "=" * 50)
print("최종 예측 수행...")
print("=" * 50)

# 예측을 위해 학습 데이터를 시간과 발전소 ID로 인덱싱하여 검색 속도 향상
train_pivot = train.pivot(index='time', columns='pv_id', values='nins')

# 예측값을 저장할 빈 컬럼 생성
test['nins'] = 0.0

# 각 테스트 데이터 행에 대해 예측 수행
predictions = []
for _, row in test.iterrows():
    time, test_pv_id = row['time'], row['pv_id']
    
    # 미리 찾아놓은 이웃 정보 가져오기
    neighbors = neighbor_map.get(test_pv_id)
    if not neighbors:
        predictions.append(0.0)
        continue
        
    neighbor_ids = neighbors['ids']
    neighbor_dists = neighbors['dists']
    
    # 현재 시간(time)에 해당하는 이웃들의 일사량(nins) 값 조회
    try:
        neighbor_nins = train_pivot.loc[time, neighbor_ids].values
    except KeyError: # 해당 시간에 데이터가 없는 경우
        predictions.append(0.0)
        continue

    # 유효한(NaN이 아닌) 이웃 데이터만 필터링
    valid_mask = ~np.isnan(neighbor_nins)
    if not np.any(valid_mask):
        predictions.append(0.0)
        continue
        
    valid_nins = neighbor_nins[valid_mask]
    valid_dists = neighbor_dists[valid_mask]
    
    # 거리의 역수를 가중치로 사용 (가까울수록 높은 가중치)
    weights = 1.0 / (valid_dists + 1e-6) # 0으로 나누는 것을 방지
    
    # 가중 평균 계산
    weighted_avg = np.average(valid_nins, weights=weights)
    predictions.append(weighted_avg)

test['nins'] = predictions
# 물리적 제약: 일사량은 음수일 수 없음
test['nins'] = test['nins'].clip(lower=0)
print("✅ 최종 예측 완료")

# ==========================================
# 검증 (Validation)
# ==========================================
print("\n" + "=" * 50)
print("검증 수행...")
print("=" * 50)

# 1. 검증 데이터 분리
#    - 학습 데이터의 일부를 검증용으로 사용. 발전소 ID를 기준으로 분리하여 데이터 유출 방지.
train_main_ids, val_ids = train_test_split(train_locations.index, test_size=0.2, random_state=42)
train_main_loc = train_locations.loc[train_main_ids]
val_loc = train_locations.loc[val_ids]

# 2. 검증용 KD-Tree 및 이웃 탐색
val_kdtree = cKDTree(train_main_loc[['coord1', 'coord2']].values)
val_dists, val_indices = val_kdtree.query(val_loc[['coord1', 'coord2']].values, k=K_NEIGHBORS)

val_neighbor_map = {}
for i, val_pv_id in enumerate(val_loc.index):
    val_neighbor_map[val_pv_id] = {
        'ids': train_main_loc.index[val_indices[i]],
        'dists': val_dists[i]
    }

# 3. 검증 데이터에 대한 예측
val_data = train[train['pv_id'].isin(val_ids)].sample(n=VALIDATION_SAMPLE_SIZE, random_state=42)
val_train_pivot = train[train['pv_id'].isin(train_main_ids)].pivot(index='time', columns='pv_id', values='nins')

val_predictions = []
for _, row in val_data.iterrows():
    time, pv_id = row['time'], row['pv_id']
    neighbors = val_neighbor_map.get(pv_id)
    if not neighbors:
        val_predictions.append(0.0)
        continue
    
    try:
        neighbor_nins = val_train_pivot.loc[time, neighbors['ids']].values
    except KeyError:
        val_predictions.append(0.0)
        continue

    valid_mask = ~np.isnan(neighbor_nins)
    if not np.any(valid_mask):
        val_predictions.append(0.0)
        continue
        
    weights = 1.0 / (neighbors['dists'][valid_mask] + 1e-6)
    weighted_avg = np.average(neighbor_nins[valid_mask], weights=weights)
    val_predictions.append(weighted_avg)

val_data['pred_nins'] = np.clip(val_predictions, 0, None)

# 4. MAE 계산
mae = mean_absolute_error(val_data['nins'], val_data['pred_nins'])
print(f"  - 검증 MAE (k={K_NEIGHBORS}): {mae:.4f}")
print("✅ 검증 완료")

# ==========================================
# 제출 파일 생성
# ==========================================
print("\n" + "=" * 50)
print("제출 파일 생성...")
print("=" * 50)

submission['nins'] = test['nins'].values
submission_filename = '3_nearest_neighbor_submission.csv'
submission.to_csv(submission_filename, index=False)

print(f"✅ 제출 파일 저장 완료: '{submission_filename}'")
print(f"  - 예측 결과 평균: {submission['nins'].mean():.4f}")
print("=" * 50)
