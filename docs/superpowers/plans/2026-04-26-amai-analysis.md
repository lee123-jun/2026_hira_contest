# AMAI 강원도 만성질환 외래 의료 공백 분석 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 강원도 18개 시군구 만성질환 외래 의료 공백을 AMAI 지수로 정량화하고, Prophet으로 공급 감소를 예측하여 2×2 정책 우선순위 매트릭스를 도출한다.

**Architecture:** `강원도_통합데이터.csv` (31,824행, 26전문과목×18시군구×68분기)를 단일 입력으로 사용. Phase 1→2→3→4 파이프라인이 각자 CSV를 `output/results/`에 저장하고 다음 Phase가 이를 입력으로 사용. `src/` 모듈은 재사용 로직, `analysis/` 스크립트는 각 Phase의 실행 진입점.

**Tech Stack:** Python 3.10+, pandas 2.x, numpy, scikit-learn, prophet, matplotlib, seaborn, scipy, plotly

---
## 커밋 확인용
## File Map

| 파일 | 역할 |
|------|------|
| `src/config.py` | 경로 상수, 만성질환 전문과목 목록, 접근성 점수 테이블 |
| `src/data_loader.py` | CSV 로드, 컬럼 정규화, 데이터 검증 |
| `src/amai.py` | 공급·수요·접근성 지수 계산, PCA 가중치, AMAI 산출 |
| `src/clustering.py` | K-Means 클러스터링, 엘보우 분석 |
| `src/forecasting.py` | Prophet 모델 학습·예측 |
| `src/visualization.py` | 모든 그래프 생성 함수 |
| `analysis/phase1_eda.py` | EDA 실행 — 현황 히트맵·분포 |
| `analysis/phase2_amai.py` | AMAI 지수 산출 + K-Means 실행 |
| `analysis/phase3_forecast.py` | Prophet 예측 실행 |
| `analysis/phase4_policy.py` | 2×2 우선순위 매트릭스 실행 |
| `run_all.py` | 전체 파이프라인 순차 실행 |
| `requirements.txt` | 의존성 목록 |

---

## Task 1: 프로젝트 셋업 (config + data_loader + requirements)

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `src/data_loader.py`

- [ ] **Step 1: requirements.txt 생성**

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
prophet>=1.1.4
scipy>=1.11.0
```

- [ ] **Step 2: src/__init__.py 생성 (빈 파일)**

```python
```

- [ ] **Step 3: src/config.py 생성**

```python
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

DATA_FILE = DATA_DIR / "강원도_통합데이터.csv"

# 만성질환 외래 관련 전문과목 (추후 실제 데이터 확인 후 보정)
CHRONIC_SPECIALTIES = [
    '내과', '가정의학과', '신경과', '정신건강의학과',
    '내분비내과', '심장내과', '호흡기내과', '소화기내과',
]

# 접근성 점수 (0=매우 취약, 1=접근 용이)
# 근거: 심평원 의료취약지 지정 현황 + 행정구역 도시/농촌 분류
# 가정: 군 지역은 기본 0.5 이하, 도시(시) 지역은 0.6 이상
ACCESS_SCORES = {
    '춘천시': 0.90, '원주시': 0.95, '강릉시': 0.88,
    '동해시': 0.75, '속초시': 0.72, '태백시': 0.65,
    '삼척시': 0.60, '홍천군': 0.45, '횡성군': 0.50,
    '영월군': 0.40, '평창군': 0.35, '정선군': 0.30,
    '철원군': 0.35, '화천군': 0.20, '양구군': 0.20,
    '인제군': 0.25, '고성군': 0.28, '양양군': 0.35,
}

GANGWON_DISTRICTS = list(ACCESS_SCORES.keys())
```

- [ ] **Step 4: src/data_loader.py 생성**

```python
import pandas as pd
from src.config import DATA_FILE, GANGWON_DISTRICTS


def load_integrated_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, encoding='cp949')
    df.columns = [
        '시군구', '연도', '분기코드', '기간',
        '전문과목', '전문의수', '고령인구비율',
        '65세이상인구', '전체인구',
    ]
    # 분기코드 "01-04" → 정수 분기 (1, 2, 3, 4)
    df['분기'] = df['분기코드'].str.extract(r'^(\d+)').astype(int)
    # Prophet 및 시계열용 날짜
    df['날짜'] = pd.PeriodIndex(
        df['연도'].astype(str) + 'Q' + df['분기'].astype(str), freq='Q'
    ).to_timestamp()
    df = df.drop(columns=['분기코드', '기간'])
    return df


def validate_data(df: pd.DataFrame) -> None:
    assert df.isnull().sum().sum() == 0, "결측값 발견"
    assert df['시군구'].nunique() == 18, f"시군구 수 오류: {df['시군구'].nunique()}"
    assert df['전문과목'].nunique() == 26, f"전문과목 수 오류: {df['전문과목'].nunique()}"
    missing = set(GANGWON_DISTRICTS) - set(df['시군구'].unique())
    assert not missing, f"누락된 시군구: {missing}"
    print(f"✅ 검증 완료: {df.shape[0]}행, {df['시군구'].nunique()}개 시군구, "
          f"{df['전문과목'].nunique()}개 전문과목")
```

- [ ] **Step 5: 패키지 설치**

```bash
pip install -r requirements.txt
```

- [ ] **Step 6: 데이터 로드 검증 실행**

```bash
cd C:/lap/hira_contest
python -c "
from src.data_loader import load_integrated_data, validate_data
df = load_integrated_data()
validate_data(df)
print(df.dtypes)
print(df.head(3))
"
```

Expected output:
```
✅ 검증 완료: 31824행, 18개 시군구, 26개 전문과목
```

---

## Task 2: Phase 1 — EDA (현황 분석)

**Files:**
- Create: `src/visualization.py`
- Create: `analysis/phase1_eda.py`

- [ ] **Step 1: src/visualization.py 생성**

```python
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def save_fig(fig: plt.Figure, name: str, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → 저장: {path}")


def plot_specialist_heatmap(
    pivot: pd.DataFrame,
    title: str,
    figures_dir: Path,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot, cmap='YlOrRd_r', ax=ax, linewidths=0.3,
                cbar_kws={'label': '인구 1000명당 전문의 수'})
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('분기')
    ax.set_ylabel('시군구')
    save_fig(fig, filename, figures_dir)


def plot_trend_lines(
    df: pd.DataFrame,
    districts: list,
    y_col: str,
    title: str,
    figures_dir: Path,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for d in districts:
        sub = df[df['시군구'] == d].sort_values('날짜')
        ax.plot(sub['날짜'], sub[y_col], label=d, linewidth=1.2)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('날짜')
    ax.set_ylabel(y_col)
    ax.legend(ncol=3, fontsize=8)
    save_fig(fig, filename, figures_dir)
```

- [ ] **Step 2: analysis/phase1_eda.py 생성**

```python
"""
Phase 1: EDA — 강원도 만성질환 외래 현황 분석
출력: output/figures/eda_*.png
     output/results/eda_summary.csv
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import pandas as pd
from src.config import CHRONIC_SPECIALTIES, FIGURES_DIR, RESULTS_DIR
from src.data_loader import load_integrated_data, validate_data
from src.visualization import plot_specialist_heatmap, plot_trend_lines, save_fig
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_integrated_data()
    validate_data(df)

    # 1. 전문과목 목록 출력 (만성질환 전문과목 확인용)
    print("\n[전문과목 목록]")
    for s in sorted(df['전문과목'].unique()):
        flag = "★" if s in CHRONIC_SPECIALTIES else " "
        print(f"  {flag} {s}")

    # 2. 만성질환 전문과목 필터 — 실제 존재 여부 확인 후 CHRONIC_SPECIALTIES 보정
    available_chronic = [s for s in CHRONIC_SPECIALTIES if s in df['전문과목'].unique()]
    print(f"\n✅ 사용 가능한 만성질환 전문과목: {available_chronic}")

    df_chronic = df[df['전문과목'].isin(available_chronic)].copy()

    # 3. 시군구×분기별 집계 (전문의 수 합산, 인구는 동일하므로 mean)
    agg = df_chronic.groupby(['시군구', '날짜']).agg(
        전문의수=('전문의수', 'sum'),
        고령인구비율=('고령인구비율', 'mean'),
        세이상인구=('65세이상인구', 'mean'),
        전체인구=('전체인구', 'mean'),
    ).reset_index()
    agg['per1000'] = agg['전문의수'] / (agg['전체인구'] / 1000)

    # 4. 히트맵: 시군구 × 연도(최근 분기)
    latest = agg.sort_values('날짜').drop_duplicates('시군구', keep='last')
    pivot_latest = agg.pivot_table(
        index='시군구', columns='날짜', values='per1000'
    )
    plot_specialist_heatmap(
        pivot_latest.iloc[:, -20:],  # 최근 5년(20분기)
        '강원도 시군구별 만성질환 전문의 수 (인구 1000명당) — 최근 5년',
        FIGURES_DIR,
        'eda_heatmap_recent5y',
    )

    # 5. 추세선: 상위 3 도시 vs 하위 3 군 비교
    by_per1000 = latest.set_index('시군구')['per1000'].sort_values(ascending=False)
    top3 = list(by_per1000.head(3).index)
    bot3 = list(by_per1000.tail(3).index)
    plot_trend_lines(
        agg, top3 + bot3, 'per1000',
        '인구 1000명당 만성질환 전문의 수 추세 (상위 3 vs 하위 3)',
        FIGURES_DIR, 'eda_trend_top_bot',
    )

    # 6. 고령화율 추세 (전체 18개)
    elderly_pivot = agg.pivot_table(index='시군구', columns='날짜', values='고령인구비율')
    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(elderly_pivot.iloc[:, -20:], cmap='Blues', ax=ax,
                cbar_kws={'label': '고령인구비율 (%)'})
    ax.set_title('강원도 시군구별 고령인구비율 — 최근 5년', fontsize=13)
    from src.visualization import save_fig
    save_fig(fig, 'eda_elderly_heatmap', FIGURES_DIR)

    # 7. 기초 통계 저장
    summary = agg.groupby('시군구')['per1000'].agg(['mean', 'min', 'max', 'std'])
    summary.columns = ['평균_per1000', '최소_per1000', '최대_per1000', '표준편차']
    summary = summary.sort_values('평균_per1000')
    summary.to_csv(RESULTS_DIR / 'eda_summary.csv', encoding='utf-8-sig')
    print(f"\n[시군구별 전문의 현황 요약]")
    print(summary.to_string())
    print(f"\n✅ Phase 1 완료 — figures: {FIGURES_DIR}, results: {RESULTS_DIR}")


if __name__ == '__main__':
    run()
```

- [ ] **Step 3: Phase 1 실행**

```bash
cd C:/lap/hira_contest
python analysis/phase1_eda.py
```

Expected output:
```
✅ 검증 완료: 31824행, 18개 시군구, 26개 전문과목
[전문과목 목록]
  ★ 가정의학과
  ...
✅ Phase 1 완료
```

> **주의:** 출력된 전문과목 목록을 보고 `src/config.py`의 `CHRONIC_SPECIALTIES`를 실제 존재하는 이름으로 수정하라. 예: '내분비내과' 대신 '내분비대사내과' 등.

- [ ] **Step 4: CHRONIC_SPECIALTIES 보정**

`analysis/phase1_eda.py`를 실행해서 출력된 전문과목 목록 중 만성질환 외래와 관련된 항목을 확인하고 `src/config.py`의 `CHRONIC_SPECIALTIES` 리스트를 실제 데이터에 존재하는 이름으로 교체.

```bash
python -c "
from src.data_loader import load_integrated_data
df = load_integrated_data()
for s in sorted(df['전문과목'].unique()): print(s)
"
```

---

## Task 3: Phase 2 — AMAI 지수 산출 + K-Means 클러스터링

**Files:**
- Create: `src/amai.py`
- Create: `src/clustering.py`
- Create: `analysis/phase2_amai.py`

- [ ] **Step 1: src/amai.py 생성**

```python
"""
AMAI (Accessible Medical Access Index) 계산 모듈

AMAI = w1 × 공급지수 + w2 × 수요지수 + w3 × 접근성지수
- 공급지수: 만성질환 전문의수 / (전체인구 / 1000)  → 정규화 [0,1]
- 수요지수: 만성질환 전문의수 / (65세이상인구 / 1000) → 정규화 [0,1]
- 접근성지수: 심평원 의료취약지 기반 사전 점수 (config.ACCESS_SCORES)
- 가중치 w1,w2,w3: PCA 1st PC 절대 로딩값을 정규화하여 사용
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from src.config import ACCESS_SCORES


def build_amai_input(agg: pd.DataFrame) -> pd.DataFrame:
    """집계된 시군구×분기 데이터를 받아 3개 원시 지표를 추가한다."""
    df = agg.copy()
    # 분모가 0인 케이스 방지 (인구 0은 없지만 방어)
    df['공급_raw'] = df['전문의수'] / (df['전체인구'] / 1000).clip(lower=0.001)
    df['수요_raw'] = df['전문의수'] / (df['세이상인구'] / 1000).clip(lower=0.001)
    df['접근성_raw'] = df['시군구'].map(ACCESS_SCORES)
    return df


def normalize_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """공급·수요 지표를 전체 데이터 기준 Min-Max 정규화한다. 접근성은 이미 [0,1]."""
    scaler = MinMaxScaler()
    df = df.copy()
    df[['공급지수', '수요지수']] = scaler.fit_transform(df[['공급_raw', '수요_raw']])
    df['접근성지수'] = df['접근성_raw']
    return df


def compute_pca_weights(df: pd.DataFrame) -> np.ndarray:
    """3개 지수에 대해 PCA 1st PC 로딩의 절대값을 정규화하여 가중치를 반환한다."""
    X = df[['공급지수', '수요지수', '접근성지수']].values
    pca = PCA(n_components=1)
    pca.fit(X)
    loadings = np.abs(pca.components_[0])
    weights = loadings / loadings.sum()
    print(f"  PCA 가중치 — 공급: {weights[0]:.3f}, 수요: {weights[1]:.3f}, 접근성: {weights[2]:.3f}")
    print(f"  PCA 1st PC 분산 설명율: {pca.explained_variance_ratio_[0]:.1%}")
    return weights


def compute_amai(df: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """AMAI 지수를 계산하고 최종 [0,1] 정규화한다."""
    df = df.copy()
    df['AMAI_raw'] = (
        weights[0] * df['공급지수']
        + weights[1] * df['수요지수']
        + weights[2] * df['접근성지수']
    )
    scaler = MinMaxScaler()
    df['AMAI'] = scaler.fit_transform(df[['AMAI_raw']])
    return df
```

- [ ] **Step 2: src/clustering.py 생성**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def elbow_analysis(X: np.ndarray, max_k: int = 8) -> list[float]:
    """k=2~max_k에 대해 inertia를 계산하여 반환한다."""
    inertias = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    return inertias


def plot_elbow(inertias: list[float], figures_dir: Path) -> None:
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = range(2, 2 + len(inertias))
    ax.plot(list(ks), inertias, 'o-', color='steelblue')
    ax.set_xlabel('클러스터 수 k')
    ax.set_ylabel('Inertia')
    ax.set_title('K-Means 엘보우 분석')
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / 'clustering_elbow.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → 저장: {figures_dir / 'clustering_elbow.png'}")


def run_kmeans(df: pd.DataFrame, features: list[str], k: int) -> pd.DataFrame:
    """시군구별 최신 분기 데이터를 클러스터링하고 cluster 컬럼을 추가한다."""
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df = df.copy()
    df['cluster'] = km.fit_predict(X_scaled)
    # AMAI 기준 오름차순으로 클러스터 레이블 재정렬 (0=고위험, k-1=저위험)
    cluster_amai_mean = df.groupby('cluster')['AMAI'].mean().sort_values()
    label_map = {old: new for new, old in enumerate(cluster_amai_mean.index)}
    df['cluster'] = df['cluster'].map(label_map)
    cluster_names = {0: '고위험', 1: '중위험', 2: '저위험', 3: '안전'}
    df['위험등급'] = df['cluster'].map({k: v for k, v in cluster_names.items() if k < 4})
    return df
```

- [ ] **Step 3: analysis/phase2_amai.py 생성**

```python
"""
Phase 2: AMAI 지수 산출 + K-Means 클러스터링
입력: output/results/eda_summary.csv (Phase 1), 통합데이터
출력: output/results/amai_by_district_quarter.csv
     output/results/amai_latest.csv (최신 분기 시군구별)
     output/figures/amai_*.png
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import CHRONIC_SPECIALTIES, FIGURES_DIR, RESULTS_DIR
from src.data_loader import load_integrated_data, validate_data
from src.amai import build_amai_input, normalize_indicators, compute_pca_weights, compute_amai
from src.clustering import elbow_analysis, plot_elbow, run_kmeans
from src.visualization import save_fig


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_integrated_data()
    validate_data(df)

    # 만성질환 전문과목 필터 + 시군구×분기 집계
    available = [s for s in CHRONIC_SPECIALTIES if s in df['전문과목'].unique()]
    df_chronic = df[df['전문과목'].isin(available)]
    agg = df_chronic.groupby(['시군구', '날짜']).agg(
        전문의수=('전문의수', 'sum'),
        고령인구비율=('고령인구비율', 'mean'),
        세이상인구=('65세이상인구', 'mean'),
        전체인구=('전체인구', 'mean'),
    ).reset_index()

    # AMAI 계산
    agg = build_amai_input(agg)
    agg = normalize_indicators(agg)
    weights = compute_pca_weights(agg)
    agg = compute_amai(agg, weights)

    # 전체 시계열 저장
    agg.to_csv(RESULTS_DIR / 'amai_by_district_quarter.csv', index=False, encoding='utf-8-sig')

    # 최신 분기 추출
    latest = agg.sort_values('날짜').drop_duplicates('시군구', keep='last').copy()

    # K-Means 엘보우 분석
    features = ['공급지수', '수요지수', '접근성지수', 'AMAI']
    inertias = elbow_analysis(latest[features].values)
    plot_elbow(inertias, FIGURES_DIR)
    print("  엘보우 분석 완료 — k=3 권장 (소수 시군구 기준)")

    # K-Means k=3 실행
    latest = run_kmeans(latest, features, k=3)
    latest.to_csv(RESULTS_DIR / 'amai_latest.csv', index=False, encoding='utf-8-sig')

    # AMAI 막대 그래프 (위험등급 색상)
    fig, ax = plt.subplots(figsize=(12, 5))
    color_map = {'고위험': '#d62728', '중위험': '#ff7f0e', '저위험': '#2ca02c'}
    for _, row in latest.sort_values('AMAI').iterrows():
        ax.bar(row['시군구'], row['AMAI'],
               color=color_map.get(row['위험등급'], '#1f77b4'))
    ax.axhline(latest['AMAI'].mean(), linestyle='--', color='gray', label='평균')
    ax.set_title('강원도 시군구별 AMAI 지수 (최신 분기)', fontsize=13)
    ax.set_ylabel('AMAI (0=최저, 1=최고)')
    ax.set_xticklabels(latest.sort_values('AMAI')['시군구'], rotation=45, ha='right')
    handles = [plt.Rectangle((0,0),1,1, color=c) for c in color_map.values()]
    ax.legend(handles, color_map.keys())
    save_fig(fig, 'amai_bar_latest', FIGURES_DIR)

    # AMAI 시계열 히트맵
    amai_pivot = agg.pivot_table(index='시군구', columns='날짜', values='AMAI')
    fig2, ax2 = plt.subplots(figsize=(18, 6))
    sns.heatmap(amai_pivot.iloc[:, -20:], cmap='RdYlGn', ax=ax2,
                vmin=0, vmax=1, cbar_kws={'label': 'AMAI'})
    ax2.set_title('강원도 시군구별 AMAI 시계열 (최근 5년)', fontsize=13)
    save_fig(fig2, 'amai_heatmap_timeseries', FIGURES_DIR)

    # 결과 출력
    print("\n[시군구별 AMAI 및 위험등급 (최신 분기)]")
    print(latest[['시군구', 'AMAI', '공급지수', '수요지수', '접근성지수', '위험등급']]
          .sort_values('AMAI').to_string(index=False))
    print(f"\n✅ Phase 2 완료")


if __name__ == '__main__':
    run()
```

- [ ] **Step 4: Phase 2 실행**

```bash
cd C:/lap/hira_contest
python analysis/phase2_amai.py
```

Expected output:
```
  PCA 가중치 — 공급: 0.xxx, 수요: 0.xxx, 접근성: 0.xxx
  엘보우 분석 완료 — k=3 권장
[시군구별 AMAI 및 위험등급]
...화천군  0.03  ...  고위험
...원주시  0.89  ...  저위험
✅ Phase 2 완료
```

---

## Task 4: Phase 3 — Prophet 공급 감소 예측

**Files:**
- Create: `src/forecasting.py`
- Create: `analysis/phase3_forecast.py`

- [ ] **Step 1: src/forecasting.py 생성**

```python
"""
Prophet 기반 전문의 수 예측 모듈
입력: 시군구×분기 시계열 (ds, y 형식)
출력: 2025~2028년 분기별 예측값 + 신뢰구간
"""
import pandas as pd
import numpy as np
from prophet import Prophet


def prepare_prophet_input(
    agg: pd.DataFrame,
    district: str,
    target_col: str = 'per1000',
) -> pd.DataFrame:
    """Prophet이 요구하는 ds, y 컬럼 형식으로 변환한다."""
    sub = agg[agg['시군구'] == district][['날짜', target_col]].copy()
    sub = sub.rename(columns={'날짜': 'ds', target_col: 'y'})
    sub = sub.sort_values('ds').reset_index(drop=True)
    return sub


def fit_and_forecast(
    prophet_df: pd.DataFrame,
    periods: int = 12,
    freq: str = 'QS',
) -> tuple[Prophet, pd.DataFrame]:
    """Prophet 모델을 학습하고 periods 분기 예측한다. (기본 3년=12분기)"""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,  # 추세 변화에 보수적 (의료인력은 급변 드묾)
    )
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return model, forecast


def extract_forecast_summary(
    forecast: pd.DataFrame,
    district: str,
) -> pd.DataFrame:
    """예측 결과에서 핵심 컬럼만 추출하고 시군구 컬럼을 추가한다."""
    cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']
    result = forecast[cols].copy()
    result['시군구'] = district
    return result
```

- [ ] **Step 2: analysis/phase3_forecast.py 생성**

```python
"""
Phase 3: Prophet 공급 감소 예측
입력: output/results/amai_by_district_quarter.csv
출력: output/results/forecast_all_districts.csv
     output/figures/forecast_*.png
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, RESULTS_DIR, GANGWON_DISTRICTS
from src.forecasting import prepare_prophet_input, fit_and_forecast, extract_forecast_summary
from src.visualization import save_fig


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    agg = pd.read_csv(RESULTS_DIR / 'amai_by_district_quarter.csv', encoding='utf-8-sig')
    agg['날짜'] = pd.to_datetime(agg['날짜'])
    agg['per1000'] = agg['전문의수'] / (agg['전체인구'] / 1000).clip(lower=0.001)

    all_forecasts = []
    risk_districts = []  # 예측 기간 내 per1000 감소 시군구

    for district in GANGWON_DISTRICTS:
        if district not in agg['시군구'].values:
            print(f"  ⚠️ {district} 데이터 없음, 건너뜀")
            continue

        prophet_df = prepare_prophet_input(agg, district, 'per1000')
        try:
            model, forecast = fit_and_forecast(prophet_df, periods=12, freq='QS')
        except Exception as e:
            print(f"  ⚠️ {district} 예측 실패: {e}")
            continue

        summary = extract_forecast_summary(forecast, district)
        all_forecasts.append(summary)

        # 예측 기간(미래 12분기) 추세 확인
        future_forecast = forecast[forecast['ds'] > prophet_df['ds'].max()]
        trend_change = future_forecast['trend'].iloc[-1] - future_forecast['trend'].iloc[0]
        if trend_change < 0:
            risk_districts.append({'시군구': district, '추세변화': round(trend_change, 4)})

        # 개별 시군구 예측 그래프
        fig, ax = plt.subplots(figsize=(10, 4))
        hist = prophet_df
        ax.plot(hist['ds'], hist['y'], 'k.', markersize=4, label='실측')
        ax.plot(forecast['ds'], forecast['yhat'], color='steelblue', label='예측')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'],
                        forecast['yhat_upper'], alpha=0.2, color='steelblue')
        ax.axvline(prophet_df['ds'].max(), linestyle='--', color='red', alpha=0.5,
                   label='예측 시작')
        ax.set_title(f'{district} — 만성질환 전문의 수 예측 (인구 1000명당)', fontsize=12)
        ax.set_ylabel('전문의/1000명')
        ax.legend(fontsize=9)
        save_fig(fig, f'forecast_{district}', FIGURES_DIR)

    # 전체 저장
    forecast_df = pd.concat(all_forecasts, ignore_index=True)
    forecast_df.to_csv(RESULTS_DIR / 'forecast_all_districts.csv',
                       index=False, encoding='utf-8-sig')

    # 감소 예측 시군구 요약
    if risk_districts:
        risk_df = pd.DataFrame(risk_districts).sort_values('추세변화')
        print("\n[예측 기간 내 공급 감소 예상 시군구]")
        print(risk_df.to_string(index=False))
        risk_df.to_csv(RESULTS_DIR / 'forecast_risk_districts.csv',
                       index=False, encoding='utf-8-sig')

    # 비교 그래프: 감소 vs 증가 시군구
    fig2, axes = plt.subplots(2, 1, figsize=(14, 8))
    for district in GANGWON_DISTRICTS[:6]:
        sub = forecast_df[forecast_df['시군구'] == district]
        if len(sub) > 0:
            axes[0].plot(sub['ds'], sub['yhat'], label=district)
    axes[0].set_title('전문의 수 예측 추이 (도시 시군구)', fontsize=12)
    axes[0].legend(ncol=3, fontsize=8)

    for district in GANGWON_DISTRICTS[-6:]:
        sub = forecast_df[forecast_df['시군구'] == district]
        if len(sub) > 0:
            axes[1].plot(sub['ds'], sub['yhat'], label=district)
    axes[1].set_title('전문의 수 예측 추이 (농촌 시군구)', fontsize=12)
    axes[1].legend(ncol=3, fontsize=8)

    plt.tight_layout()
    save_fig(fig2, 'forecast_comparison', FIGURES_DIR)

    print(f"\n✅ Phase 3 완료 — {len(all_forecasts)}개 시군구 예측")


if __name__ == '__main__':
    run()
```

- [ ] **Step 3: Phase 3 실행**

```bash
cd C:/lap/hira_contest
python analysis/phase3_forecast.py
```

Expected output:
```
[예측 기간 내 공급 감소 예상 시군구]
시군구  추세변화
양구군  -0.0312
...
✅ Phase 3 완료 — 18개 시군구 예측
```

---

## Task 5: Phase 4 — 2×2 정책 우선순위 매트릭스

**Files:**
- Create: `analysis/phase4_policy.py`

정책 매트릭스 축 정의:
- X축: 현재 AMAI (낮을수록 공백 심각)
- Y축: 예측 추세 (음수일수록 향후 악화)

4사분면 해석:
- 1사분면 (AMAI 낮음 + 예측 악화): **즉시 개입 필요** — 정책 최우선 순위
- 2사분면 (AMAI 높음 + 예측 악화): **모니터링 강화** — 선제적 관리
- 3사분면 (AMAI 낮음 + 예측 개선): **구조 지원** — 회복 지원
- 4사분면 (AMAI 높음 + 예측 개선): **현상 유지** — 모범 사례 분석

- [ ] **Step 1: analysis/phase4_policy.py 생성**

```python
"""
Phase 4: 2×2 정책 우선순위 매트릭스
입력: output/results/amai_latest.csv
     output/results/forecast_all_districts.csv
출력: output/results/policy_matrix.csv
     output/figures/policy_matrix.png
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.config import RESULTS_DIR, FIGURES_DIR
from src.visualization import save_fig


QUADRANT_LABELS = {
    (False, True):  ('즉시 개입', '#d62728'),   # AMAI 낮음 + 악화
    (True, True):   ('모니터링 강화', '#ff7f0e'), # AMAI 높음 + 악화
    (False, False): ('구조 지원', '#9467bd'),    # AMAI 낮음 + 개선
    (True, True):   ('현상 유지', '#2ca02c'),    # AMAI 높음 + 개선
}

# (AMAI_above_median, trend_improving) → label, color
QUADRANT_MAP = {
    (False, False): ('즉시 개입',     '#d62728'),
    (False, True):  ('구조 지원',     '#9467bd'),
    (True,  False): ('모니터링 강화', '#ff7f0e'),
    (True,  True):  ('현상 유지',     '#2ca02c'),
}


def compute_trend_slope(forecast_df: pd.DataFrame, district: str) -> float:
    """예측 기간(미래)의 선형 추세 기울기를 반환한다."""
    sub = forecast_df[forecast_df['시군구'] == district].copy()
    sub = sub.sort_values('ds')
    # 마지막 12개 행(예측 기간)
    future = sub.tail(12)
    if len(future) < 2:
        return 0.0
    x = np.arange(len(future))
    slope = np.polyfit(x, future['yhat'].values, 1)[0]
    return float(slope)


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    amai_latest = pd.read_csv(RESULTS_DIR / 'amai_latest.csv', encoding='utf-8-sig')
    forecast_df = pd.read_csv(RESULTS_DIR / 'forecast_all_districts.csv', encoding='utf-8-sig')
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

    # 예측 추세 기울기 계산
    slopes = {d: compute_trend_slope(forecast_df, d) for d in amai_latest['시군구']}
    amai_latest['예측추세'] = amai_latest['시군구'].map(slopes)

    # 매트릭스 분류
    amai_median = amai_latest['AMAI'].median()
    amai_latest['AMAI_above'] = amai_latest['AMAI'] >= amai_median
    amai_latest['trend_improving'] = amai_latest['예측추세'] >= 0

    amai_latest['정책우선순위'] = amai_latest.apply(
        lambda r: QUADRANT_MAP[(r['AMAI_above'], r['trend_improving'])][0], axis=1
    )
    amai_latest['우선순위_색'] = amai_latest.apply(
        lambda r: QUADRANT_MAP[(r['AMAI_above'], r['trend_improving'])][1], axis=1
    )

    # 결과 저장
    output_cols = ['시군구', 'AMAI', '공급지수', '수요지수', '접근성지수',
                   '위험등급', '예측추세', '정책우선순위']
    policy_df = amai_latest[output_cols].sort_values(
        ['정책우선순위', 'AMAI']
    )
    policy_df.to_csv(RESULTS_DIR / 'policy_matrix.csv', index=False, encoding='utf-8-sig')

    # 2×2 산점도 매트릭스
    fig, ax = plt.subplots(figsize=(10, 8))
    for _, row in amai_latest.iterrows():
        ax.scatter(row['AMAI'], row['예측추세'],
                   color=row['우선순위_색'], s=120, zorder=5)
        ax.annotate(row['시군구'], (row['AMAI'], row['예측추세']),
                    textcoords='offset points', xytext=(6, 4), fontsize=8)

    # 중앙선 (중앙값)
    ax.axvline(amai_median, linestyle='--', color='gray', alpha=0.6)
    ax.axhline(0, linestyle='--', color='gray', alpha=0.6)

    # 사분면 레이블
    ax.text(amai_median * 0.3, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else -0.001,
            '① 즉시 개입', fontsize=11, color='#d62728', fontweight='bold')
    ax.text(amai_median * 1.2, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else -0.001,
            '② 모니터링 강화', fontsize=11, color='#ff7f0e', fontweight='bold')
    ax.text(amai_median * 0.3, ax.get_ylim()[0] * 0.9 if ax.get_ylim()[0] < 0 else 0.001,
            '③ 구조 지원', fontsize=11, color='#9467bd', fontweight='bold')
    ax.text(amai_median * 1.2, ax.get_ylim()[0] * 0.9 if ax.get_ylim()[0] < 0 else 0.001,
            '④ 현상 유지', fontsize=11, color='#2ca02c', fontweight='bold')

    ax.set_xlabel('AMAI 지수 (현재 의료 접근성, 높을수록 양호)', fontsize=11)
    ax.set_ylabel('예측 추세 기울기 (양수=공급 증가, 음수=감소)', fontsize=11)
    ax.set_title('강원도 시군구별 정책 우선순위 매트릭스\n(X: 현재 AMAI, Y: 미래 공급 추세)', fontsize=13)

    legend_elements = [
        mpatches.Patch(color=c, label=l)
        for (_, _), (l, c) in QUADRANT_MAP.items()
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    save_fig(fig, 'policy_matrix', FIGURES_DIR)

    # 콘솔 요약
    print("\n[정책 우선순위 매트릭스 결과]")
    for priority in ['즉시 개입', '모니터링 강화', '구조 지원', '현상 유지']:
        subset = policy_df[policy_df['정책우선순위'] == priority]
        if len(subset) > 0:
            districts = ', '.join(subset['시군구'].tolist())
            print(f"\n  [{priority}] ({len(subset)}개 시군구)")
            print(f"  → {districts}")

    print(f"\n✅ Phase 4 완료 — 결과: {RESULTS_DIR / 'policy_matrix.csv'}")


if __name__ == '__main__':
    run()
```

- [ ] **Step 2: Phase 4 실행**

```bash
cd C:/lap/hira_contest
python analysis/phase4_policy.py
```

Expected output:
```
[정책 우선순위 매트릭스 결과]

  [즉시 개입] (N개 시군구)
  → 화천군, 양구군, 인제군, ...

  [모니터링 강화] (N개 시군구)
  → ...

✅ Phase 4 완료
```

---

## Task 6: 전체 파이프라인 통합 (run_all.py)

**Files:**
- Create: `run_all.py`

- [ ] **Step 1: run_all.py 생성**

```python
"""
전체 분석 파이프라인 순차 실행
Phase 1 → Phase 2 → Phase 3 → Phase 4
"""
import time

def run_phase(name: str, module_path: str) -> None:
    import importlib.util, sys
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    start = time.time()
    spec = importlib.util.spec_from_file_location("phase", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run()
    print(f"  ⏱️ 소요시간: {time.time() - start:.1f}초")


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    run_phase("Phase 1: EDA", "analysis/phase1_eda.py")
    run_phase("Phase 2: AMAI 지수 산출", "analysis/phase2_amai.py")
    run_phase("Phase 3: Prophet 예측", "analysis/phase3_forecast.py")
    run_phase("Phase 4: 정책 매트릭스", "analysis/phase4_policy.py")

    print("\n" + "="*60)
    print("  ✅ 전체 파이프라인 완료")
    print("  결과: output/figures/ 및 output/results/")
    print("="*60)
```

- [ ] **Step 2: 전체 파이프라인 실행**

```bash
cd C:/lap/hira_contest
python run_all.py
```

Expected output:
```
============================================================
  Phase 1: EDA
============================================================
✅ Phase 1 완료
  ⏱️ 소요시간: X.Xs

...

============================================================
  ✅ 전체 파이프라인 완료
  결과: output/figures/ 및 output/results/
============================================================
```

---

## Self-Review

### 스펙 커버리지 체크

| 요구사항 | 구현된 Task |
|---------|------------|
| EDA — 현황 지도(Choropleth 대체: 히트맵) | Task 2 |
| AMAI 지수 산출 | Task 3 |
| PCA 가중치 | Task 3 (src/amai.py `compute_pca_weights`) |
| K-Means 클러스터링 위험 그룹 분류 | Task 3 (src/clustering.py) |
| 공급 감소 예측 — Prophet | Task 4 |
| 2×2 우선순위 매트릭스 | Task 5 |
| 정책 권고안 출력 | Task 5 (콘솔 요약 + CSV) |
| 인구 1000명당 전문의 수 계산 | Task 2, 4 |
| 고령인구비율 수요 지표 | Task 3 (수요지수) |
| 접근성지수 (의료취약지 기반) | Task 3 (config.ACCESS_SCORES) |

### 가정 명시 (공모전 발표 시 필수 언급)
1. **CHRONIC_SPECIALTIES**: 실제 데이터 전문과목명 확인 후 Task 2 Step 4에서 수동 보정 필요
2. **ACCESS_SCORES**: 심평원 공식 의료취약지 지정 자료 미입수로 행정구역(시/군) + 지리적 근접성 기반 전문가 추정치 사용. 논문 값으로 대체 가능.
3. **Prophet freq='QS'**: 분기 시작일 기준. 실제 날짜 변환 결과 확인 필요.
4. **K-Means k=3**: 엘보우 분석 후 최적 k로 조정.
