import pandas as pd
from src.config import DATA_FILE, GANGWON_DISTRICTS


def load_integrated_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, encoding='cp949')
    df.columns = [
        '시군구', '연도', '분기코드', '기간',
        '전문과목', '전문의수', '고령인구비율',
        '65세이상인구', '전체인구',
    ]
    # 분기코드 "01-04" -> 정수 분기 (1, 2, 3, 4)
    df['분기'] = df['분기코드'].str.extract(r'^(\d+)').astype(int)
    # 시계열용 날짜 (분기 시작일)
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
    print(
        f"[OK] 검증 완료: {df.shape[0]}행, {df['시군구'].nunique()}개 시군구, "
        f"{df['전문과목'].nunique()}개 전문과목"
    )
