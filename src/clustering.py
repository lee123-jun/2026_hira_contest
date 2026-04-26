import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def elbow_analysis(X: np.ndarray, max_k: int = 8) -> list:
    inertias = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    return inertias


def plot_elbow(inertias: list, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = list(range(2, 2 + len(inertias)))
    ax.plot(ks, inertias, 'o-', color='steelblue')
    ax.set_xlabel('클러스터 수 k')
    ax.set_ylabel('Inertia')
    ax.set_title('K-Means 엘보우 분석')
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / 'clustering_elbow.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> 저장: {figures_dir / 'clustering_elbow.png'}")


def run_kmeans(df: pd.DataFrame, features: list, k: int) -> pd.DataFrame:
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
    df['위험등급'] = df['cluster'].map({ki: v for ki, v in cluster_names.items() if ki < k})
    return df
