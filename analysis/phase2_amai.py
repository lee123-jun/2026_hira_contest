"""
Phase 2: AMAI 지수 산출 + K-Means 클러스터링
입력: 강원도_통합데이터.csv
출력: output/results/amai_by_district_quarter.csv
     output/results/amai_latest.csv
     output/figures/amai_*.png, clustering_elbow.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

    available = [s for s in CHRONIC_SPECIALTIES if s in df['전문과목'].unique()]
    if not available:
        print("[WARN]  CHRONIC_SPECIALTIES 미매칭. src/config.py 수정 후 재실행.")
        return

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

    agg.to_csv(RESULTS_DIR / 'amai_by_district_quarter.csv', index=False, encoding='utf-8-sig')

    # 최신 분기
    latest = agg.sort_values('날짜').drop_duplicates('시군구', keep='last').copy()

    # 엘보우 분석
    features = ['공급지수', '수요지수', '접근성지수', 'AMAI']
    inertias = elbow_analysis(latest[features].values)
    plot_elbow(inertias, FIGURES_DIR)

    # K-Means k=3
    latest = run_kmeans(latest, features, k=3)
    latest.to_csv(RESULTS_DIR / 'amai_latest.csv', index=False, encoding='utf-8-sig')

    # AMAI 막대 그래프 (위험등급 색상)
    color_map = {'고위험': '#d62728', '중위험': '#ff7f0e', '저위험': '#2ca02c'}
    sorted_latest = latest.sort_values('AMAI')
    fig, ax = plt.subplots(figsize=(12, 5))
    for _, row in sorted_latest.iterrows():
        ax.bar(row['시군구'], row['AMAI'],
               color=color_map.get(row['위험등급'], '#1f77b4'))
    ax.axhline(latest['AMAI'].mean(), linestyle='--', color='gray', label='평균')
    ax.set_title('강원도 시군구별 AMAI 지수 (최신 분기)', fontsize=13)
    ax.set_ylabel('AMAI (0=최저, 1=최고)')
    ax.set_xticklabels(sorted_latest['시군구'], rotation=45, ha='right')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_map.values()]
    ax.legend(handles, color_map.keys())
    save_fig(fig, 'amai_bar_latest', FIGURES_DIR)

    # AMAI 시계열 히트맵
    amai_pivot = agg.pivot_table(index='시군구', columns='날짜', values='AMAI')
    fig2, ax2 = plt.subplots(figsize=(18, 6))
    sns.heatmap(amai_pivot.iloc[:, -20:], cmap='RdYlGn', ax=ax2,
                vmin=0, vmax=1, cbar_kws={'label': 'AMAI'})
    ax2.set_title('강원도 시군구별 AMAI 시계열 (최근 5년)', fontsize=13)
    save_fig(fig2, 'amai_heatmap_timeseries', FIGURES_DIR)

    print("\n[시군구별 AMAI 및 위험등급 (최신 분기)]")
    print(
        latest[['시군구', 'AMAI', '공급지수', '수요지수', '접근성지수', '위험등급']]
        .sort_values('AMAI')
        .to_string(index=False)
    )
    print(f"\n[OK] Phase 2 완료")


if __name__ == '__main__':
    run()
