# Architecture Decision Record — AMAI 강원도 만성질환 외래 의료 공백 분석

## 프로젝트 개요
건강보험심사평가원(HIRA) 2026 빅데이터 창업 공모전 출품작.
강원도 18개 시군구 만성질환 외래 의료 공백을 AMAI 지수로 정량화하고, Prophet으로 공급 감소를 예측하여 2x2 정책 우선순위 매트릭스를 도출하는 분석 파이프라인.

## 데이터
- 입력: `data/강원도_통합데이터.csv` (31,824행, 결측 0건)
  - 26개 전문과목 x 18개 시군구 x 68개 분기 (2009Q1~2025Q4)
  - 컬럼: 시군구, 연도, 분기코드, 기간, 전문과목, 전문의수, 고령인구비율, 65세이상인구, 전체인구
- 인코딩: cp949 (로드 시), utf-8-sig (저장 시)

## 아키텍처 결정

### 1. 파이프라인 구조 (Phase 직렬 체인)
Phase 1 (EDA) -> Phase 2 (AMAI) -> Phase 3 (Prophet) -> Phase 4 (Policy)
각 Phase는 이전 Phase의 CSV 출력을 `output/results/`에서 읽어 사용.
전체 실행: `python run_all.py`

### 2. 만성질환 전문과목 선정
실제 데이터의 26개 전문과목 중 5개 선정 (src/config.py CHRONIC_SPECIALTIES):
- 내과, 가정의학과, 신경과, 정신건강의학과, 재활의학과
- 근거: 고혈압·당뇨·심뇌혈관·정신질환 1차 외래 담당 과목
- 내분비내과/심장내과/호흡기내과는 본 데이터에서 내과로 통합되어 있음

### 3. AMAI 공식
AMAI = w1 * 공급지수 + w2 * 수요지수 + w3 * 접근성지수
- 공급지수: 만성질환 전문의수 / (전체인구 / 1000), Min-Max 정규화
- 수요지수: 만성질환 전문의수 / (65세이상인구 / 1000), Min-Max 정규화
- 접근성지수: 심평원 의료취약지 기반 사전 점수 (src/config.py ACCESS_SCORES)
- 가중치: PCA 1st PC 절대 로딩값 정규화 → 공급 0.313 / 수요 0.362 / 접근성 0.326
- PCA 1st PC 분산 설명율: 90.2%

### 4. 접근성 지수 가정 (중요)
심평원 공식 의료취약지 데이터 미확보로 행정구역(시/군) + 지리적 근접성 기반 전문가 추정치 사용.
시 지역: 0.60~0.95, 군 지역: 0.20~0.50. 공식 데이터로 교체 시 정확도 향상 가능.

### 5. K-Means 클러스터링
- k=3 (고위험/중위험/저위험), 엘보우 분석으로 결정
- 특징: 공급지수, 수요지수, 접근성지수, AMAI 4개 변수 StandardScaler 후 클러스터링
- 클러스터 레이블: AMAI 평균 오름차순 재정렬 (0=고위험)

### 6. Prophet 예측
- 18개 시군구 개별 모델, 12분기(3년) 예측
- changepoint_prior_scale=0.05 (의료인력 급변 드묾)
- freq='QS' (분기 시작일 기준)

### 7. 정책 매트릭스 분류 기준
- X축: AMAI >= 중앙값 여부
- Y축: Prophet 예측 추세 기울기 >= 0 여부
- 즉시 개입: AMAI 낮음 + 기울기 음수 (6개 시군구)
- 모니터링 강화: AMAI 높음 + 기울기 음수 (2개 시군구)
- 구조 지원: AMAI 낮음 + 기울기 양수 (3개 시군구)
- 현상 유지: AMAI 높음 + 기울기 양수 (7개 시군구)

## 파일 구조
```
src/
  config.py          # 경로 상수, CHRONIC_SPECIALTIES, ACCESS_SCORES
  data_loader.py     # CSV 로드 + 검증
  amai.py            # AMAI 지수 계산 (PCA 가중치)
  clustering.py      # K-Means 엘보우 + 클러스터링
  forecasting.py     # Prophet 학습/예측
  visualization.py   # 그래프 공통 함수
analysis/
  phase1_eda.py      # EDA (히트맵, 추세선)
  phase2_amai.py     # AMAI + K-Means
  phase3_forecast.py # Prophet 18개 시군구
  phase4_policy.py   # 2x2 정책 매트릭스
output/
  figures/           # 그래프 26개 (PNG)
  results/           # CSV 6개
run_all.py           # 전체 파이프라인
```

## 출력물
- figures: eda_heatmap_recent5y, eda_elderly_heatmap, eda_trend_top_bot,
           amai_bar_latest, amai_heatmap_timeseries, clustering_elbow,
           forecast_{시군구} x18, forecast_comparison, policy_matrix
- results: eda_summary.csv, amai_by_district_quarter.csv, amai_latest.csv,
           forecast_all_districts.csv, forecast_risk_districts.csv, policy_matrix.csv
