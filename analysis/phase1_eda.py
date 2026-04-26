"""
Phase 1: EDA - 강원도 만성질환 외래 현황 분석
출력: output/figures/eda_*.png
     output/results/eda_summary.csv
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
from src.visualization import plot_specialist_heatmap, plot_trend_lines, save_fig


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_integrated_data()
    validate_data(df)

    # 전문과목 목록 출력 - CHRONIC_SPECIALTIES 보정용
    print("\n[전문과목 목록] ([*] = CHRONIC_SPECIALTIES에 포함됨)")
    for s in sorted(df['전문과목'].unique()):
        flag = "[*]" if s in CHRONIC_SPECIALTIES else " "
        print(f"  {flag} {s}")

    available_chronic = [s for s in CHRONIC_SPECIALTIES if s in df['전문과목'].unique()]
    print(f"\n[OK] 매칭된 만성질환 전문과목 ({len(available_chronic)}개): {available_chronic}")

    if not available_chronic:
        print("[WARN]  CHRONIC_SPECIALTIES가 실제 데이터와 일치하지 않습니다.")
        print("    위 전문과목 목록을 확인하고 src/config.py를 수정하세요.")
        return

    df_chronic = df[df['전문과목'].isin(available_chronic)].copy()

    # 시군구x분기 집계
    agg = df_chronic.groupby(['시군구', '날짜']).agg(
        전문의수=('전문의수', 'sum'),
        고령인구비율=('고령인구비율', 'mean'),
        세이상인구=('65세이상인구', 'mean'),
        전체인구=('전체인구', 'mean'),
    ).reset_index()
    agg['per1000'] = agg['전문의수'] / (agg['전체인구'] / 1000).clip(lower=0.001)

    # 히트맵: 최근 5년(20분기)
    pivot = agg.pivot_table(index='시군구', columns='날짜', values='per1000')
    plot_specialist_heatmap(
        pivot.iloc[:, -20:],
        '강원도 시군구별 만성질환 전문의 수 (인구 1000명당) - 최근 5년',
        FIGURES_DIR,
        'eda_heatmap_recent5y',
    )

    # 추세선: 상위 3 vs 하위 3 시군구
    latest = agg.sort_values('날짜').drop_duplicates('시군구', keep='last')
    by_per1000 = latest.set_index('시군구')['per1000'].sort_values(ascending=False)
    top3 = list(by_per1000.head(3).index)
    bot3 = list(by_per1000.tail(3).index)
    plot_trend_lines(
        agg, top3 + bot3, 'per1000',
        '인구 1000명당 만성질환 전문의 수 추세 (상위 3 vs 하위 3)',
        FIGURES_DIR, 'eda_trend_top_bot',
    )

    # 고령화율 히트맵
    elderly_pivot = agg.pivot_table(index='시군구', columns='날짜', values='고령인구비율')
    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(elderly_pivot.iloc[:, -20:], cmap='Blues', ax=ax,
                cbar_kws={'label': '고령인구비율 (%)'})
    ax.set_title('강원도 시군구별 고령인구비율 - 최근 5년', fontsize=13)
    save_fig(fig, 'eda_elderly_heatmap', FIGURES_DIR)

    # 기초 통계 저장
    summary = agg.groupby('시군구')['per1000'].agg(['mean', 'min', 'max', 'std'])
    summary.columns = ['평균_per1000', '최소_per1000', '최대_per1000', '표준편차']
    summary = summary.sort_values('평균_per1000')
    summary.to_csv(RESULTS_DIR / 'eda_summary.csv', encoding='utf-8-sig')

    print(f"\n[시군구별 전문의 현황 요약]")
    print(summary.to_string())
    print(f"\n[OK] Phase 1 완료 - figures: {FIGURES_DIR}")


if __name__ == '__main__':
    run()
