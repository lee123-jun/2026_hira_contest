"""
Prophet 기반 전문의 수 예측 모듈
"""
import pandas as pd
from prophet import Prophet


def prepare_prophet_input(
    agg: pd.DataFrame,
    district: str,
    target_col: str = 'per1000',
) -> pd.DataFrame:
    sub = agg[agg['시군구'] == district][['날짜', target_col]].copy()
    sub = sub.rename(columns={'날짜': 'ds', target_col: 'y'})
    sub = sub.sort_values('ds').reset_index(drop=True)
    return sub


def fit_and_forecast(
    prophet_df: pd.DataFrame,
    periods: int = 12,
    freq: str = 'QS',
) -> tuple:
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return model, forecast


def extract_forecast_summary(
    forecast: pd.DataFrame,
    district: str,
) -> pd.DataFrame:
    cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']
    result = forecast[cols].copy()
    result['시군구'] = district
    return result
