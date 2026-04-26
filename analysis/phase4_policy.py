"""
Phase 4: 2x2 정책 우선순위 매트릭스
입력: output/results/amai_latest.csv
     output/results/forecast_all_districts.csv
출력: output/results/policy_matrix.csv
     output/figures/policy_matrix.png

매트릭스 축:
  X: 현재 AMAI (낮을수록 공백 심각)
  Y: 예측 추세 기울기 (음수=공급 감소)

4사분면:
  ① 즉시 개입    - AMAI 낮음 + 공급 감소 예측
  ② 모니터링 강화 - AMAI 높음 + 공급 감소 예측
  ③ 구조 지원    - AMAI 낮음 + 공급 증가 예측
  ④ 현상 유지    - AMAI 높음 + 공급 증가 예측
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.config import RESULTS_DIR, FIGURES_DIR
from src.visualization import save_fig

# (AMAI_above_median, trend_improving) -> (레이블, 색상)
QUADRANT_MAP = {
    (False, False): ('① 즉시 개입',     '#d62728'),
    (False, True):  ('③ 구조 지원',     '#9467bd'),
    (True,  False): ('② 모니터링 강화', '#ff7f0e'),
    (True,  True):  ('④ 현상 유지',     '#2ca02c'),
}


def compute_trend_slope(forecast_df: pd.DataFrame, district: str) -> float:
    sub = forecast_df[forecast_df['시군구'] == district].sort_values('ds')
    future = sub.tail(12)
    if len(future) < 2:
        return 0.0
    x = np.arange(len(future))
    slope = float(np.polyfit(x, future['yhat'].values, 1)[0])
    return slope


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    amai_latest = pd.read_csv(RESULTS_DIR / 'amai_latest.csv', encoding='utf-8-sig')
    forecast_df = pd.read_csv(
        RESULTS_DIR / 'forecast_all_districts.csv', encoding='utf-8-sig'
    )
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

    slopes = {d: compute_trend_slope(forecast_df, d) for d in amai_latest['시군구']}
    amai_latest['예측추세'] = amai_latest['시군구'].map(slopes)

    amai_median = amai_latest['AMAI'].median()
    amai_latest['AMAI_above'] = amai_latest['AMAI'] >= amai_median
    amai_latest['trend_improving'] = amai_latest['예측추세'] >= 0

    amai_latest['정책우선순위'] = amai_latest.apply(
        lambda r: QUADRANT_MAP[(r['AMAI_above'], r['trend_improving'])][0], axis=1
    )
    amai_latest['우선순위_색'] = amai_latest.apply(
        lambda r: QUADRANT_MAP[(r['AMAI_above'], r['trend_improving'])][1], axis=1
    )

    output_cols = [
        '시군구', 'AMAI', '공급지수', '수요지수', '접근성지수',
        '위험등급', '예측추세', '정책우선순위',
    ]
    policy_df = amai_latest[output_cols].sort_values(['정책우선순위', 'AMAI'])
    policy_df.to_csv(RESULTS_DIR / 'policy_matrix.csv', index=False, encoding='utf-8-sig')

    # 2x2 산점도
    fig, ax = plt.subplots(figsize=(11, 9))

    for _, row in amai_latest.iterrows():
        ax.scatter(row['AMAI'], row['예측추세'],
                   color=row['우선순위_색'], s=150, zorder=5, edgecolors='white', linewidths=0.5)
        ax.annotate(
            row['시군구'], (row['AMAI'], row['예측추세']),
            textcoords='offset points', xytext=(7, 4), fontsize=9,
        )

    ax.axvline(amai_median, linestyle='--', color='gray', alpha=0.6, linewidth=1)
    ax.axhline(0, linestyle='--', color='gray', alpha=0.6, linewidth=1)

    # 사분면 레이블 배치
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_pad = (xlim[1] - xlim[0]) * 0.03
    y_pad = (ylim[1] - ylim[0]) * 0.04
    quadrant_texts = [
        (xlim[0] + x_pad, ylim[0] + y_pad,     '① 즉시 개입',     '#d62728'),
        (xlim[0] + x_pad, ylim[1] - y_pad * 2, '③ 구조 지원',     '#9467bd'),
        (amai_median + x_pad, ylim[0] + y_pad,  '② 모니터링 강화', '#ff7f0e'),
        (amai_median + x_pad, ylim[1] - y_pad * 2, '④ 현상 유지', '#2ca02c'),
    ]
    for x, y, label, color in quadrant_texts:
        ax.text(x, y, label, fontsize=11, color=color, fontweight='bold', alpha=0.7)

    ax.set_xlabel('AMAI 지수 (현재 의료 접근성, 높을수록 양호)', fontsize=11)
    ax.set_ylabel('예측 추세 기울기 (양수=공급 증가, 음수=감소)', fontsize=11)
    ax.set_title(
        '강원도 시군구별 정책 우선순위 매트릭스\n(X: 현재 AMAI, Y: 미래 공급 추세)',
        fontsize=13,
    )

    legend_elements = [
        mpatches.Patch(color=c, label=l)
        for (_, _), (l, c) in QUADRANT_MAP.items()
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    save_fig(fig, 'policy_matrix', FIGURES_DIR)

    # 콘솔 요약
    print("\n[정책 우선순위 매트릭스 결과]")
    for priority in ['① 즉시 개입', '② 모니터링 강화', '③ 구조 지원', '④ 현상 유지']:
        subset = policy_df[policy_df['정책우선순위'] == priority]
        if len(subset) > 0:
            districts = ', '.join(subset['시군구'].tolist())
            print(f"\n  [{priority}] ({len(subset)}개 시군구)")
            print(f"  -> {districts}")

    print(f"\n[OK] Phase 4 완료 - 결과: {RESULTS_DIR / 'policy_matrix.csv'}")


if __name__ == '__main__':
    run()
