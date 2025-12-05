
# 2025 OIBC Team CHIDIES

태양광 발전량 추정(Photovoltaic Generation Forecast) 대회에서 사용한 모델 코드와 실행 스크립트 모음입니다. 이 저장소는 발표 자료와 함께 모델 학습, 예측, 제출 파일 생성까지 재현할 수 있도록 구성되어 있습니다.

## 주요 목적
- 대회에서 사용된 여러 모델(기본 베이스라인, 보간/감쇠/국소 모델, 앙상블 등)을 제공하여 성능을 비교하고 재현할 수 있게 합니다.
- 각 스크립트는 가능한 한 독립적으로 실행되며, 최종 예측을 위해 지정된 실행 순서를 따르세요.

## 저장소 구조

- `final_codes_TEAM_CHIDIES/`
	- `0_baseline.py` : 기본 LightGBM 베이스라인 예시
	- `1_1_preprocess_data.py`, `1_2_interpolate_model.py` : 1단계 전처리 및 보간 모델
	- `2_1_preprocess_data.py`, `2_2_attenuation_model.py` : 2단계 전처리 및 감쇠(attenuation) 모델
	- `3_nearest_neighbor_model.py` : 최근접 이웃 기반 예측 모델
	- `4_local_IDW_model.py` : 국소 IDW(가중 거리 보간) 모델
	- `5_1_generate_trend_predictions.py`, `5_2_regression_IDW_model.py` : 트렌드 예측 및 회귀 기반 IDW
	- `6_final_ensemble_model.py` : 여러 모델을 합친 최종 앙상블 모델
	- `info.md` : 모델 실행 순서와 간단 설명
- `requirements.txt` : 파이썬 의존성 목록
- `README.md` : 이 파일
- `LICENSE` : 프로젝트 라이선스(MIT)
- `태양광 발전량 추정 전략_TEAM CHIDIES.pdf` : 발표자료 (한글)

## 요구사항(Prerequisites)
- Python 3.8 이상 권장
- 가상환경(venv, conda 등)을 사용하여 종속성을 격리해 주세요.

## 설치 및 환경 설정

예시 (venv 사용):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 데이터 준비
- 이 저장소에서는 데이터 파일을 포함하지 않습니다(대회 규정 및 개인정보 등 이유).
- 로컬에 다음 파일들을 준비하세요:
	- `train.csv` (학습 데이터)
	- `test.csv` (테스트 / 예측 대상 데이터)
	- `submission_sample.csv` (제출 형식 샘플)

- 데이터 파일 위치:
	- 기본적으로 스크립트는 프로젝트 루트(이 레포지토리의 루트)를 기준으로 파일을 참조합니다. 필요 시 경로를 수정하세요.

## 사용 방법 — 스크립트 실행 순서 및 설명

일반적으로 전처리 스크립트를 먼저 실행하고, 모델별로 학습/예측 스크립트를 실행한 뒤 최종 앙상블 스크립트를 실행합니다. `final_codes_TEAM_CHIDIES/info.md`에 권장 순서가 있습니다.

권장 실행 순서(최종 모델 결과 재현):

1. `final_codes_TEAM_CHIDIES/2_1_preprocess_data.py` — 전처리 (데이터 정리/기본 특성 생성)
2. `final_codes_TEAM_CHIDIES/4_local_IDW_model.py` — 국소 IDW 모델 수행
3. `final_codes_TEAM_CHIDIES/5_1_generate_trend_predictions.py` — 트렌드 예측 생성
4. `final_codes_TEAM_CHIDIES/5_2_regression_IDW_model.py` — 회귀 기반 IDW 모델 수행
5. `final_codes_TEAM_CHIDIES/6_final_ensemble_model.py` — 최종 앙상블

각 스크립트는 `info.md` 또는 파일 상단에 포함된 주석에 더 자세한 실행 방법(입력/출력 파일 명, 옵션 등)을 안내합니다.

## 스크립트별 요약
- `0_baseline.py` — LightGBM을 이용한 빠른 baseline 예시. 학습 데이터로 검증 및 제출 파일 생성까지 포함.
- `1_*` — 초기 전처리 및 보간 관련 스크립트
- `2_*` — 추가 전처리 및 물리 기반 감쇠(attenuation) 모델
- `3_*` — 최근접 이웃 기반 모델(설명: k-NN 스타일)
- `4_*`, `5_*` — 국소/회귀 IDW 및 트렌드 보정 스텝
- `6_final_ensemble_model.py` — 위 모델들을 조합한 앙상블

## 연락 & 참고
- 작성자: 김수민
- 대회 & 데이터 출처: OIBC(2025)

감사합니다.

