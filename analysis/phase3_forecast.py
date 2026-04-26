"""
Phase 3: Prophet 공급 감소 예측
입력: output/results/amai_by_district_quarter.csv
출력: output/results/forecast_all_districts.csv
     output/results/forecast_risk_districts.csv
     output/figures/forecast_*.png
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

from src.config import FIGURES_DIR, RESULTS_DIR, GANGWON_DISTRICTS
from src.forecasting import prepare_prophet_input, fit_and_forecast, extract_forecast_summary
from src.visualization import save_fig


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    agg = pd.read_csv(RESULTS_DIR / 'amai_by_district_quarter.csv', encoding='utf-8-sig')
    agg['날짜'] = pd.to_datetime(agg['날짜'])
    agg['per1000'] = agg['전문의수'] / (agg['전체인구'] / 1000).clip(lower=0.001)

    all_forecasts = []
    risk_districts = []

    for district in GANGWON_DISTRICTS:
        if district not in agg['시군구'].values:
            print(f"  [WARN] {district} 데이터 없음, 건너뜀")
            continue

        prophet_df = prepare_prophet_input(agg, district, 'per1000')
        try:
            _, forecast = fit_and_forecast(prophet_df, periods=12, freq='QS')
        except Exception as e:
            print(f"  [WARN] {district} 예측 실패: {e}")
            continue

        summary = extract_forecast_summary(forecast, district)
        all_forecasts.append(summary)

        # 미래 12분기 추세 방향 확인
        future = forecast[forecast['ds'] > prophet_df['ds'].max()]
        if len(future) > 1:
            trend_change = future['trend'].iloc[-1] - future['trend'].iloc[0]
            if trend_change < 0:
                risk_districts.append({'시군구': district, '추세변화': round(trend_change, 4)})

        # 개별 예측 그래프
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(prophet_df['ds'], prophet_df['y'], 'k.', markersize=4, label='실측')
        ax.plot(forecast['ds'], forecast['yhat'], color='steelblue', label='예측')
        ax.fill_between(
            forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
            alpha=0.2, color='steelblue',
        )
        ax.axvline(prophet_df['ds'].max(), linestyle='--', color='red',
                   alpha=0.5, label='예측 시작')
        ax.set_title(f'{district} - 만성질환 전문의 수 예측 (인구 1000명당)', fontsize=12)
        ax.set_ylabel('전문의/1000명')
        ax.legend(fontsize=9)
        save_fig(fig, f'forecast_{district}', FIGURES_DIR)

    forecast_df = pd.concat(all_forecasts, ignore_index=True)
    forecast_df.to_csv(
        RESULTS_DIR / 'forecast_all_districts.csv', index=False, encoding='utf-8-sig'
    )

    if risk_districts:
        risk_df = pd.DataFrame(risk_districts).sort_values('추세변화')
        print("\n[예측 기간 내 공급 감소 예상 시군구]")
        print(risk_df.to_string(index=False))
        risk_df.to_csv(
            RESULTS_DIR / 'forecast_risk_districts.csv', index=False, encoding='utf-8-sig'
        )

    # 비교 그래프: 도시 vs 농촌
    fig2, axes = plt.subplots(2, 1, figsize=(14, 8))
    city_districts = [d for d in GANGWON_DISTRICTS if d.endswith('시')]
    county_districts = [d for d in GANGWON_DISTRICTS if d.endswith('군')]
    for district in city_districts:
        sub = forecast_df[forecast_df['시군구'] == district]
        if len(sub) > 0:
            axes[0].plot(sub['ds'], sub['yhat'], label=district)
    axes[0].set_title('전문의 수 예측 추이 (시 지역)', fontsize=12)
    axes[0].legend(ncol=4, fontsize=8)

    for district in county_districts:
        sub = forecast_df[forecast_df['시군구'] == district]
        if len(sub) > 0:
            axes[1].plot(sub['ds'], sub['yhat'], label=district)
    axes[1].set_title('전문의 수 예측 추이 (군 지역)', fontsize=12)
    axes[1].legend(ncol=4, fontsize=8)

    plt.tight_layout()
    save_fig(fig2, 'forecast_comparison', FIGURES_DIR)

    print(f"\n[OK] Phase 3 완료 - {len(all_forecasts)}개 시군구 예측")


if __name__ == '__main__':
    run()
