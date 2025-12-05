# ==========================================
# 라이브러리 임포트
# ==========================================
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ==========================================
# 데이터 로드
# ==========================================
print("=" * 50)
print("데이터 로딩 시작...")
print("=" * 50)

# 학습, 테스트, 제출 샘플 데이터 로드
try:
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    submission = pd.read_csv('submission_sample.csv')
    print("✅ 데이터 로딩 완료")
except FileNotFoundError as e:
    print(f"❌ 파일 로딩 오류: {e}. 스크립트 실행 위치를 확인하세요.")
    exit()

print(f"Train 데이터 형태: {train.shape}")
print(f"Test 데이터 형태: {test.shape}")

# ==========================================
# 기본 전처리
# ==========================================
print("\n" + "=" * 50)
print("기본 전처리 시작...")
print("=" * 50)

# 1. 시간 데이터 변환
# 'time' 컬럼을 datetime 객체로 변환하여 시간 관련 특성을 다룰 수 있도록 함
print("  - 'time' 컬럼을 datetime으로 변환 중...")
train['time'] = pd.to_datetime(train['time'])
test['time'] = pd.to_datetime(test['time'])

# 2. 학습에 사용할 특성 선택
# 타겟 변수('nins')와 식별자('time', 'pv_id', 'type'), 참고용 변수('energy')를 제외한 모든 컬럼을 특성으로 사용
features = [col for col in train.columns if col not in ['time', 'pv_id', 'type', 'energy', 'nins']]
print(f"  - 학습에 사용될 특성 개수: {len(features)}")

# 3. 결측치 처리 (단순 Backward Fill)
# 기상 데이터는 시간 순서에 따라 관측되므로, 가장 가까운 미래의 데이터로 현재의 결측치를 채움 (bfill)
# 발전소(pv_id)별로 그룹화하여 다른 발전소의 데이터가 섞이지 않도록 함
print("  - 결측치를 backward fill 방식으로 채우는 중...")
train[features] = train.groupby('pv_id')[features].transform(lambda x: x.bfill())
test[features] = test.groupby('pv_id')[features].transform(lambda x: x.bfill())

# 4. 전처리 후 남은 결측치 제거
# bfill로도 채워지지 않는 데이터(주로 각 발전소의 맨 처음 데이터)가 있을 경우 해당 행을 제거
train = train.dropna(subset=features)
print(f"  - 결측치 처리 후 Train 데이터 형태: {train.shape}")
print("✅ 기본 전처리 완료")

# ==========================================
# 모델 학습 및 검증 (Validation)
# ==========================================
print("\n" + "=" * 50)
print("모델 학습 및 검증 시작...")
print("=" * 50)

# 1. 학습/검증 데이터 분리
# 모델의 일반화 성능을 확인하기 위해 학습 데이터를 80%의 학습셋과 20%의 검증셋으로 분리
X_train, X_val, y_train, y_val = train_test_split(
    train[features], 
    train['nins'], 
    test_size=0.2, 
    random_state=42 # 결과 재현을 위한 시드 고정
)
print(f"  - 학습 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")

# 2. 모델 정의 및 학습
# LightGBM 회귀 모델 사용. random_state로 재현성 확보, verbose=-1로 학습 과정 로그 출력 안함
model = LGBMRegressor(random_state=42, verbose=-1)
print("  - LightGBM 모델 학습 중...")
model.fit(X_train, y_train)

# 3. 검증 데이터 예측 및 평가
print("  - 검증 데이터로 성능 평가 중...")
val_pred = model.predict(X_val)

# 물리적 제약 조건 적용: 일사량은 음수일 수 없으므로 0보다 작은 예측값은 0으로 조정
val_pred[val_pred < 0] = 0

# MAE(Mean Absolute Error) 계산
mae = mean_absolute_error(y_val, val_pred)
print(f"\n[검증 성능]")
print(f"  - MAE (Mean Absolute Error): {mae:.4f}")
print("✅ 모델 학습 및 검증 완료")

# ==========================================
# 최종 예측 및 제출 파일 생성
# ==========================================
print("\n" + "=" * 50)
print("최종 예측 및 제출 파일 생성 시작...")
print("=" * 50)

# 1. 전체 학습 데이터로 모델 재학습
# 더 많은 데이터를 사용하여 모델의 성능을 극대화하기 위해 전체 학습 데이터로 다시 학습
print("  - 전체 학습 데이터로 모델 재학습 중...")
final_model = LGBMRegressor(random_state=42, verbose=-1)
final_model.fit(train[features], train['nins'])

# 2. 테스트 데이터 예측
print("  - 테스트 데이터로 최종 예측 수행 중...")
test_pred = final_model.predict(test[features])

# 물리적 제약 조건 적용
test_pred[test_pred < 0] = 0

# 3. 제출 파일 생성
submission['nins'] = test_pred
submission_filename = '0_baseline_submission.csv'
submission.to_csv(submission_filename, index=False)

print(f"\n✅ 최종 제출 파일 저장 완료: '{submission_filename}'")
print("=" * 50)