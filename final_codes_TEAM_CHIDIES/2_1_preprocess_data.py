# ==========================================
# 라이브러리 임포트
# ==========================================
import pandas as pd
import numpy as np
import os
import gc

# ==========================================
# 파일 경로 및 설정
# ==========================================
# 입력 파일
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

# 최종 출력 파일
PROCESSED_TRAIN_FILE = '2_processed_train_attenuation.parquet'
PROCESSED_TEST_FILE = '2_processed_test_attenuation.parquet'

# ==========================================
# 특성 공학 함수
# ==========================================

def calculate_theoretical_max_irradiance_vectorized(times, latitude=36.35, longitude=127.38):
    """
    벡터화된 이론상 최대 일사량 계산 함수
    - times: pandas Series (datetime objects)
    """
    # 태양 상수
    SOLAR_CONSTANT = 1367.0
    
    # 시간 요소 추출
    day_of_year = times.dt.dayofyear
    decimal_hour = times.dt.hour + times.dt.minute / 60.0
    
    # 1. 태양 적위 (Solar Declination)
    declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
    
    # 2. 시간각 (Hour Angle)
    hour_angle = 15 * (decimal_hour - 12)
    
    # 3. 태양 고도각 (Solar Altitude)
    lat_rad = np.radians(latitude)
    dec_rad = np.radians(declination)
    hour_rad = np.radians(hour_angle)
    
    sin_altitude = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_rad)
    
    # 4. 대기권 밖 일사량
    distance_correction = 1 + 0.033 * np.cos(np.radians(360 * day_of_year / 365))
    max_irradiance = SOLAR_CONSTANT * distance_correction * sin_altitude
    
    # 음수값(지평선 아래)은 0으로 처리
    max_irradiance = np.maximum(0.0, max_irradiance)
    
    return max_irradiance.astype('float32')

def add_time_features(df):
    """시간 관련 기본 특성 및 주기적 특성 추가"""
    print("   - 시간 기반 특성 추가...")
    df['hour'] = df['time'].dt.hour.astype('uint8')
    df['day_of_year'] = df['time'].dt.dayofyear.astype('uint16')
    df['month'] = df['time'].dt.month.astype('uint8')
    df['day_of_week'] = df['time'].dt.dayofweek.astype('uint8')
    
    # 시간을 sin/cos 변환하여 주기성을 모델이 잘 학습하도록 함
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24).astype('float16')
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24).astype('float16')
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365).astype('float16')
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365).astype('float16')
    return df

def add_solar_features(df):
    """태양 위치와 관련된 특성 추가 (감쇠율 모델의 핵심) - 청크 처리 및 벡터화 적용"""
    print("   - 태양 관련 특성 추가 (청크 단위 처리)...")
    
    # 결과 저장을 위한 빈 컬럼 생성 (float32로 초기화)
    n_rows = len(df)
    df['theoretical_max_irradiance'] = np.zeros(n_rows, dtype='float32')
    df['solar_altitude_sin'] = np.zeros(n_rows, dtype='float16')
    if 'nins' in df.columns:
        df['irradiance_attenuation_rate'] = np.zeros(n_rows, dtype='float16')
    
    chunk_size = 500000 # 50만개씩 처리
    total_chunks = (n_rows + chunk_size - 1) // chunk_size
    
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_rows)
        
        if i % 5 == 0 or i == total_chunks - 1:
            print(f"     - 진행: {i+1}/{total_chunks} 청크 처리 중 ({start_idx}~{end_idx})")
            
        # 인덱스 슬라이싱을 사용하여 해당 청크의 time 컬럼 가져오기
        chunk_times = df['time'].iloc[start_idx:end_idx]
        
        # 1. 이론상 최대 일사량 계산 (벡터화 함수 호출)
        max_irradiance = calculate_theoretical_max_irradiance_vectorized(chunk_times)
        df.iloc[start_idx:end_idx, df.columns.get_loc('theoretical_max_irradiance')] = max_irradiance
        
        # 2. 태양 고도각 sin 값
        SOLAR_CONSTANT = 1367.0
        day_of_year = chunk_times.dt.dayofyear
        distance_correction = 1 + 0.033 * np.cos(np.radians(360 * day_of_year / 365))
        
        solar_alt_sin = np.where(
            max_irradiance > 0,
            max_irradiance / (SOLAR_CONSTANT * distance_correction),
            0.0
        ).clip(0, 1).astype('float16')
        df.iloc[start_idx:end_idx, df.columns.get_loc('solar_altitude_sin')] = solar_alt_sin
        
        # 3. 일사량 감쇠율 계산
        if 'nins' in df.columns:
            chunk_nins = df['nins'].iloc[start_idx:end_idx].values
            attenuation = np.where(
                max_irradiance > 0,
                chunk_nins / max_irradiance,
                0.0
            ).clip(0, 1.5).astype('float16')
            df.iloc[start_idx:end_idx, df.columns.get_loc('irradiance_attenuation_rate')] = attenuation
            
    return df

def add_weather_interaction_features(df):
    """기본적인 기상 변수들의 상호작용 특성 추가"""
    print("   - 기상 상호작용 특성 추가...")
    # RuntimeWarning 방지를 위해 nanmean 사용 시 예외 처리 고려 가능하나, 여기서는 기본 유지
    with np.errstate(invalid='ignore'): # 빈 슬라이스 경고 억제
        df['temp_mean'] = np.nanmean([df['temp_a'], df['temp_b']], axis=0).astype('float16')
        df['cloud_mean'] = np.nanmean([df['cloud_a'], df['cloud_b']], axis=0).astype('float16')
        df['wind_spd_mean'] = np.nanmean([df['wind_spd_a'], df['wind_spd_b']], axis=0).astype('float16')
    
    df['temp_humidity'] = (df['temp_mean'] * df['humidity']).astype('float16') # 온도와 습도의 조합
    df['cloud_temp'] = (df['cloud_mean'] * df['temp_mean']).astype('float16') # 구름과 온도의 조합
    df['press_diff'] = (df['ground_press'] - df['pressure']).astype('float16') # 기압 차이
    df['temp_range'] = (df['temp_max'] - df['temp_min']).astype('float16') # 일교차
    df['total_precip'] = (df['rain'] + df['snow']).astype('float16') # 총 강수량
    return df

def add_advanced_weather_features(df):
    """물리적 의미를 고려한 고급 기상 특성 추가"""
    print("   - 고급 기상 특성 추가...")
    
    # 풍향(0~360도)을 sin/cos으로 변환하여 원형 특성으로 처리
    df['wind_dir_a_sin'] = np.sin(np.radians(df['wind_dir_a'])).astype('float16')
    df['wind_dir_a_cos'] = np.cos(np.radians(df['wind_dir_a'])).astype('float16')
    
    # 구름량, 습도, 일교차의 비선형 관계를 표현하기 위해 다항 특성 추가
    df['cloud_mean_sq'] = (df['cloud_mean'] ** 2).astype('float16')
    df['humidity_sq'] = (df['humidity'] ** 2).astype('float16')
    df['temp_range_sq'] = (df['temp_range'] ** 2).astype('float16')
    
    # 구름량과 태양 고도의 상호작용 (고도가 낮을 때 구름의 영향이 더 큼)
    df['cloud_x_sol_alt'] = (df['cloud_mean'] * df['solar_altitude_sin']).astype('float16')
    
    # 이슬점과 온도의 차이 (안개/결로 가능성 지표)
    df['dew_point_spread'] = (df['temp_mean'] - df['dew_point']).astype('float16')
    
    # 시정(가시거리)의 역수 (대기 혼탁도 지표로 사용)
    df['extinction_proxy'] = (1 / (df['vis'] + 0.001)).astype('float16')
    
    return df

def interpolate_missing_values(df, dataset_name='data'):
    """결측치를 선형 보간법으로 채우는 함수"""
    print(f"   - {dataset_name} 데이터 결측치 보간 중...")
    
    features_to_interpolate = [col for col in df.columns if col not in ['time', 'pv_id', 'type', 'energy', 'nins']]
    print(f"     - 보간 대상 특성 수: {len(features_to_interpolate)}")
    
    pv_ids = df['pv_id'].unique()
    n_ids = len(pv_ids)
    chunk_size = 10 # 한 번에 처리할 발전소 수
    
    print(f"     총 {n_ids}개 발전소를 {chunk_size}개씩 묶어 처리합니다.")
    
    for i in range(0, n_ids, chunk_size):
        chunk_ids = pv_ids[i:i + chunk_size]
        current_step = i // chunk_size + 1
        total_steps = (n_ids + chunk_size - 1) // chunk_size
        
        print(f"       [{current_step}/{total_steps}] 발전소 {chunk_ids[0]} ~ {chunk_ids[-1]} 처리 중... ({len(chunk_ids)}개)")
        
        mask = df['pv_id'].isin(chunk_ids)
        
        # 필요한 컬럼만 복사하여 처리
        cols_needed = features_to_interpolate + ['pv_id']
        subset = df.loc[mask, cols_needed].copy()
        
        # float16으로 인한 interpolate/bfill 오류 방지를 위해 float32로 변환
        for col in features_to_interpolate:
            if subset[col].dtype == 'float16':
                subset[col] = subset[col].astype('float32')
        
        # Groupby transform 수행
        # 참고: Groupby 후 transform은 인덱스를 유지함
        subset[features_to_interpolate] = subset.groupby('pv_id')[features_to_interpolate].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both').bfill().ffill()
        )
        
        # 원본 데이터프레임에 업데이트 (다시 float16으로 들어갈 수 있음)
        # dtypes 호환성 경고가 뜰 수 있으나, 값 할당 자체는 수행됨
        df.loc[mask, features_to_interpolate] = subset[features_to_interpolate]
        
        # 메모리 정리
        del subset
        gc.collect()
        
    return df

# ==========================================
# 메인 전처리 실행 함수
# ==========================================
def process_train():
    """Train 데이터 전처리 및 저장"""
    print("\n" + "=" * 70)
    print("Step 1~4: Train 데이터 전처리")
    print("=" * 70)

    if not os.path.exists(TRAIN_FILE):
        print(f"❌ 오류: {TRAIN_FILE} 파일을 찾을 수 없습니다.")
        return

    print(f"   - Loading {TRAIN_FILE}...")
    # Train은 기본적으로 float32로 로드 (안정성 확보)
    cols_to_optimize = [col for col in pd.read_csv(TRAIN_FILE, nrows=0).columns if 'temp' in col or 'cloud' in col]
    dtype_map = {col: 'float32' for col in cols_to_optimize}
    
    train = pd.read_csv(TRAIN_FILE, dtype=dtype_map)
    
    print("   - 시간 컬럼 datetime으로 변환...")
    train['time'] = pd.to_datetime(train['time'])

    print("\n[Train 데이터 특성 공학]")
    train = add_time_features(train)
    train = add_solar_features(train) # 감쇠율 계산 포함
    train = add_weather_interaction_features(train)
    train = add_advanced_weather_features(train)
    
    print(f"\n   - 특성 추가 후 Train 형태: {train.shape}")

    print("\n[Train 데이터 결측치 처리 (선형 보간)]")
    train = interpolate_missing_values(train, 'Train')
    
    print("\n[Train 데이터 저장]")
    print(f"   - Saving to {PROCESSED_TRAIN_FILE}...")
    train.to_parquet(PROCESSED_TRAIN_FILE, engine='pyarrow', compression='snappy')
    print(f"   ✅ Saved: {PROCESSED_TRAIN_FILE}")

    # 메모리 정리
    del train
    gc.collect()
    print("🧹 Train 데이터 메모리 정리 완료")

def process_test():
    """Test 데이터 전처리 및 저장"""
    print("\n" + "=" * 70)
    print("Step 1~4: Test 데이터 전처리")
    print("=" * 70)

    if not os.path.exists(TEST_FILE):
        print(f"❌ 오류: {TEST_FILE} 파일을 찾을 수 없습니다.")
        return

    print(f"   - Loading {TEST_FILE}...")
    
    # [수정된 부분] float16 로드 에러 방지를 위한 로직
    # 1. 최적화할 컬럼 식별
    cols_to_optimize = [col for col in pd.read_csv(TEST_FILE, nrows=0).columns if 'temp' in col or 'cloud' in col]
    
    # 2. float32로 먼저 읽기 (pd.read_csv는 float16 직접 로드 불가)
    dtype_map = {col: 'float32' for col in cols_to_optimize}
    test = pd.read_csv(TEST_FILE, dtype=dtype_map)
    
    # 3. 로드 후 메모리 최적화를 위해 float16으로 변환
    print("   - 메모리 최적화 중 (float32 -> float16)...")
    for col in cols_to_optimize:
        test[col] = test[col].astype('float16')
    
    print("   - 시간 컬럼 datetime으로 변환...")
    test['time'] = pd.to_datetime(test['time'])

    print("\n[Test 데이터 특성 공학]")
    test = add_time_features(test)
    test = add_solar_features(test)
    test = add_weather_interaction_features(test)
    test = add_advanced_weather_features(test)
    
    print(f"\n   - 특성 추가 후 Test 형태: {test.shape}")

    print("\n[Test 데이터 결측치 처리 (선형 보간)]")
    test = interpolate_missing_values(test, 'Test')

    print("\n[Test 데이터 저장]")
    print(f"   - Saving to {PROCESSED_TEST_FILE}...")
    test.to_parquet(PROCESSED_TEST_FILE, engine='pyarrow', compression='snappy')
    print(f"   ✅ Saved: {PROCESSED_TEST_FILE}")

    # 메모리 정리
    del test
    gc.collect()
    print("🧹 Test 데이터 메모리 정리 완료")

def run_preprocessing():
    """전체 데이터 전처리 파이프라인을 실행"""
    
    print("=" * 70)
    print("2단계 모델 - 데이터 전처리 시작 (감쇠율 모델용)")
    print("=" * 70)
    
    # Train 처리
    process_train()
    
    # Test 처리
    process_test()
    
    print("\n" + "=" * 70)
    print("✅ 모든 전처리 완료!")
    print("=" * 70)
    
    print("\n💡 다음 단계: `2_2_attenuation_model.py`를 실행하여 감쇠율 모델 학습을 진행하세요.")

# ==========================================
# 스크립트 실행
# ==========================================
if __name__ == "__main__":
    # 전처리된 파일이 있으면 덮어쓸지 물어봄
    if os.path.exists(PROCESSED_TRAIN_FILE) or os.path.exists(PROCESSED_TEST_FILE):
        user_input = input("⚠️  이미 전처리된 파일이 존재합니다. 덮어쓰시겠습니까? (y/n): ").strip().lower()
        if user_input == 'y':
            run_preprocessing()
        else:
            print("✋ 전처리를 취소했습니다.")
    else:
        run_preprocessing()