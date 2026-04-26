"""
전체 분석 파이프라인 순차 실행
Phase 1 -> Phase 2 -> Phase 3 -> Phase 4
"""
import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_phase(name: str, module_path: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    start = time.time()
    spec = importlib.util.spec_from_file_location("phase", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run()
    print(f"  [TIME]  소요시간: {time.time() - start:.1f}초")


if __name__ == '__main__':
    base = Path(__file__).parent / "analysis"
    run_phase("Phase 1: EDA",              str(base / "phase1_eda.py"))
    run_phase("Phase 2: AMAI 지수 산출",   str(base / "phase2_amai.py"))
    run_phase("Phase 3: Prophet 예측",     str(base / "phase3_forecast.py"))
    run_phase("Phase 4: 정책 매트릭스",    str(base / "phase4_policy.py"))

    print(f"\n{'=' * 60}")
    print("  [OK] 전체 파이프라인 완료")
    print("  결과: output/figures/ 및 output/results/")
    print(f"{'=' * 60}")
