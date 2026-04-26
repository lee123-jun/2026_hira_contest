"""
AMAI (Accessible Medical Access Index) 계산 모듈

AMAI = w1 x 공급지수 + w2 x 수요지수 + w3 x 접근성지수
- 공급지수: 만성질환 전문의수 / (전체인구 / 1000)        -> 정규화 [0,1]
- 수요지수: 만성질환 전문의수 / (65세이상인구 / 1000)    -> 정규화 [0,1]
- 접근성지수: 심평원 의료취약지 기반 사전 점수 (config.ACCESS_SCORES, 이미 [0,1])
- 가중치 w1,w2,w3: PCA 1st PC 절대 로딩값을 정규화하여 사용
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from src.config import ACCESS_SCORES


def build_amai_input(agg: pd.DataFrame) -> pd.DataFrame:
    df = agg.copy()
    df['공급_raw'] = df['전문의수'] / (df['전체인구'] / 1000).clip(lower=0.001)
    df['수요_raw'] = df['전문의수'] / (df['세이상인구'] / 1000).clip(lower=0.001)
    df['접근성_raw'] = df['시군구'].map(ACCESS_SCORES)
    return df


def normalize_indicators(df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    df = df.copy()
    df[['공급지수', '수요지수']] = scaler.fit_transform(df[['공급_raw', '수요_raw']])
    df['접근성지수'] = df['접근성_raw']
    return df


def compute_pca_weights(df: pd.DataFrame) -> np.ndarray:
    X = df[['공급지수', '수요지수', '접근성지수']].values
    pca = PCA(n_components=1)
    pca.fit(X)
    loadings = np.abs(pca.components_[0])
    weights = loadings / loadings.sum()
    print(
        f"  PCA 가중치 - 공급: {weights[0]:.3f}, "
        f"수요: {weights[1]:.3f}, 접근성: {weights[2]:.3f}"
    )
    print(f"  PCA 1st PC 분산 설명율: {pca.explained_variance_ratio_[0]:.1%}")
    return weights


def compute_amai(df: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df['AMAI_raw'] = (
        weights[0] * df['공급지수']
        + weights[1] * df['수요지수']
        + weights[2] * df['접근성지수']
    )
    scaler = MinMaxScaler()
    df['AMAI'] = scaler.fit_transform(df[['AMAI_raw']])
    return df
