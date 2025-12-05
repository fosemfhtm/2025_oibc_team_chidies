import pandas as pd
import numpy as np
import os
import gc

# ==========================================
# 전처리 설정
# ==========================================
# 입력 파일
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

# 출력 파일
PROCESSED_TRAIN_FILE = '1_processed_train_interpolate.parquet'
PROCESSED_TEST_FILE = '1_processed_test_interpolate.parquet'

# ==========================================
# 특성 공학 함수 정의
# ==========================================
def add_time_features(df):
    """시간 기반 특성 추가"""
    print("  - 시간 기반 특성 추가 중...")
    df['hour'] = df['time'].dt.hour
    df['day_of_year'] = df['time'].dt.dayofyear
    df['month'] = df['time'].dt.month
    df['day_of_week'] = df['time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 주기적 특성 (sin/cos 변환)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def add_weather_interaction_features(df):
    """기상 상호작용 특성 추가"""
    print("  - 기상 상호작용 특성 추가 중...")
    # 평균 온도
    temp_a = df['temp_a'].values
    temp_b = df['temp_b'].values
    df['temp_mean'] = np.nanmean([temp_a, temp_b], axis=0)
    
    # 평균 구름량
    cloud_a = df['cloud_a'].values
    cloud_b = df['cloud_b'].values
    df['cloud_mean'] = np.nanmean([cloud_a, cloud_b], axis=0)
    
    # 평균 풍속
    wind_a = df['wind_spd_a'].values
    wind_b = df['wind_spd_b'].values
    df['wind_spd_mean'] = np.nanmean([wind_a, wind_b], axis=0)
    
    # 온도와 습도의 상호작용
    df['temp_humidity'] = df['temp_mean'] * df['humidity']
    
    # 구름량과 온도의 상호작용
    df['cloud_temp'] = df['cloud_mean'] * df['temp_mean']
    
    # 기압 차이
    df['press_diff'] = df['ground_press'] - df['pressure']
    
    # 온도 범위
    df['temp_range'] = df['temp_max'] - df['temp_min']
    
    # 체감온도 차이
    df['feel_temp_diff'] = df['real_feel_temp'] - df['temp_mean']
    
    # 강수 관련
    df['total_precip'] = df['rain'] + df['snow']
    
    return df

def interpolate_missing_values(df, dataset_name='data'):
    """결측치 보간 처리 (청크 단위 처리로 속도/메모리 균형)"""
    print(f"  - {dataset_name} 결측치 보간 중...")
    
    features_to_interpolate = [col for col in df.columns 
                               if col not in ['time', 'pv_id', 'type', 'energy', 'nins']]
    
    print(f"    보간 대상 특성 수: {len(features_to_interpolate)}")
    
    pv_ids = df['pv_id'].unique()
    n_ids = len(pv_ids)
    chunk_size = 10 # 한 번에 처리할 발전소 수 (메모리와 속도 균형)
    
    print(f"    총 {n_ids}개 발전소를 {chunk_size}개씩 묶어 처리합니다.")
    
    for i in range(0, n_ids, chunk_size):
        chunk_ids = pv_ids[i:i + chunk_size]
        current_step = i // chunk_size + 1
        total_steps = (n_ids + chunk_size - 1) // chunk_size
        
        print(f"      [{current_step}/{total_steps}] 발전소 {chunk_ids[0]} ~ {chunk_ids[-1]} 처리 중... ({len(chunk_ids)}개)")
        
        mask = df['pv_id'].isin(chunk_ids)
        
        # 필요한 컬럼만 복사하여 처리 (pv_id 포함)
        cols_needed = features_to_interpolate + ['pv_id']
        subset = df.loc[mask, cols_needed].copy()
        
        # Groupby transform 수행
        subset[features_to_interpolate] = subset.groupby('pv_id')[features_to_interpolate].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both').bfill().ffill()
        )
        
        # 원본 데이터프레임에 업데이트
        df.loc[mask, features_to_interpolate] = subset[features_to_interpolate]
        
        # 메모리 정리
        del subset
        gc.collect()
        
    return df

# ==========================================
# 메인 전처리 함수
# ==========================================
# ==========================================
# 메인 전처리 함수 (메모리 최적화 버전)
# ==========================================
def process_train():
    """Train 데이터 전처리 및 저장"""
    print("\n" + "=" * 70)
    print("Step 1 & 2 & 3: Train 데이터 전처리")
    print("=" * 70)

    if not os.path.exists(TRAIN_FILE):
        print(f"❌ 오류: {TRAIN_FILE} 파일을 찾을 수 없습니다.")
        return

    print(f"  - Loading {TRAIN_FILE}...")
    train = pd.read_csv(TRAIN_FILE)
    print(f"\n  원본 Train shape: {train.shape}")

    print("\n  - 시간 컬럼을 datetime으로 변환 중...")
    train['time'] = pd.to_datetime(train['time'])

    print("\n[Train 데이터 특성 공학]")
    train = add_time_features(train)
    train = add_weather_interaction_features(train)
    print("✅ Train 특성 공학 완료")

    print(f"\n  특성 추가 후 Train shape: {train.shape}")

    print("\n[Train 데이터 결측치 처리 (선형 보간)]")
    train = interpolate_missing_values(train, 'Train')

    # 최종 NaN 확인 생략 (사용자 요청)
    # train_nan_count = train.drop(columns=['energy', 'nins']).isna().sum().sum()
    # print(f"\n  보간 후 Train 최종 NaN 개수: {train_nan_count}")

    # if train_nan_count > 0:
    #     print("❌ 오류: Train 데이터 보간 후에도 NaN 값이 남아있습니다.")
    #     return

    print("\n[Train 데이터 저장]")
    print(f"  - Saving {PROCESSED_TRAIN_FILE}...")
    train.to_parquet(PROCESSED_TRAIN_FILE, engine='pyarrow', compression='snappy')
    file_size_train = os.path.getsize(PROCESSED_TRAIN_FILE) / 1024**2
    print(f"  ✅ Saved: {PROCESSED_TRAIN_FILE} ({file_size_train:.2f} MB)")

    # 메모리 정리
    del train
    gc.collect()
    print("🧹 Train 데이터 메모리 정리 완료")

def process_test():
    """Test 데이터 전처리 및 저장"""
    print("\n" + "=" * 70)
    print("Step 1 & 2 & 3: Test 데이터 전처리")
    print("=" * 70)

    if not os.path.exists(TEST_FILE):
        print(f"❌ 오류: {TEST_FILE} 파일을 찾을 수 없습니다.")
        return

    print(f"  - Loading {TEST_FILE}...")
    test = pd.read_csv(TEST_FILE)
    print(f"\n  원본 Test shape: {test.shape}")

    print("\n  - 시간 컬럼을 datetime으로 변환 중...")
    test['time'] = pd.to_datetime(test['time'])

    print("\n[Test 데이터 특성 공학]")
    test = add_time_features(test)
    test = add_weather_interaction_features(test)
    print("✅ Test 특성 공학 완료")

    print(f"\n  특성 추가 후 Test shape: {test.shape}")

    print("\n[Test 데이터 결측치 처리 (선형 보간)]")
    test = interpolate_missing_values(test, 'Test')

    # 최종 NaN 확인 생략 (사용자 요청)
    # test_nan_count = test.isna().sum().sum()
    # print(f"  보간 후 Test 최종 NaN 개수: {test_nan_count}")

    # if test_nan_count > 0:
    #     print("❌ 오류: Test 데이터 보간 후에도 NaN 값이 남아있습니다.")
    #     return

    print("\n[Test 데이터 저장]")
    print(f"  - Saving {PROCESSED_TEST_FILE}...")
    test.to_parquet(PROCESSED_TEST_FILE, engine='pyarrow', compression='snappy')
    file_size_test = os.path.getsize(PROCESSED_TEST_FILE) / 1024**2
    print(f"  ✅ Saved: {PROCESSED_TEST_FILE} ({file_size_test:.2f} MB)")

    # 메모리 정리
    del test
    gc.collect()
    print("🧹 Test 데이터 메모리 정리 완료")

def preprocess_data():
    """데이터 전처리 메인 함수"""
    
    print("=" * 70)
    print("1단계 모델 - 데이터 전처리 시작")
    print("=" * 70)
    
    # 전처리된 파일이 이미 존재하는지 확인
    if os.path.exists(PROCESSED_TRAIN_FILE) and os.path.exists(PROCESSED_TEST_FILE):
        print("\n⚠️  전처리된 파일이 이미 존재합니다:")
        print(f"  - {PROCESSED_TRAIN_FILE}")
        print(f"  - {PROCESSED_TEST_FILE}")
        
        user_input = input("\n덮어쓰시겠습니까? (y/n): ").strip().lower()
        if user_input != 'y':
            print("\n전처리를 취소합니다.")
            return
        print("\n기존 파일을 덮어씁니다...\n")

    # Train 처리
    process_train()
    
    # Test 처리
    process_test()
    
    print("\n" + "=" * 70)
    print("✅ 모든 전처리 완료!")
    print("=" * 70)
    
    print("\n💡 다음 단계:")
    print("  - `1_interpolate_model.py` 를 실행하여 모델 학습을 진행하세요.")
    print("  - 이 스크립트에서 생성된 전처리 파일이 자동으로 로드됩니다.")

# ==========================================
# 실행
# ==========================================
if __name__ == "__main__":
    try:
        preprocess_data()
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
