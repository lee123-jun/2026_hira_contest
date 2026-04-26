import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
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
    print(f"  -> 저장: {path}")


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
