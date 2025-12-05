"""
최종 앙상블 스크립트
- 4_local_IDW_submission (가중치 3) + 5_2_regression_IDW_submission (가중치 1)
- 블렌딩 결과에 196/194 미세 스케일 적용
"""

import os
import numpy as np
import pandas as pd

WEIGHT_LOCAL_IDW = 3
WEIGHT_REG_IDW = 1
SCALE_FACTOR = 196 / 194
OUTPUT_FILE = "6_final_ensemble_submission.csv"


def load_csv_with_fallback(filename: str) -> pd.DataFrame:
    """CWD 우선, 없으면 스크립트 경로에서 CSV를 읽는다."""
    if os.path.exists(filename):
        return pd.read_csv(filename)

    script_dir = os.path.dirname(__file__)
    fallback_path = os.path.join(script_dir, filename)
    if os.path.exists(fallback_path):
        return pd.read_csv(fallback_path)

    raise FileNotFoundError(f"Cannot find '{filename}' in CWD or script directory.")


def main() -> None:
    print("=" * 70)
    print("최종 앙상블: 4_local_IDW (x3) + 5_2_regression_IDW (x1), 196/194 스케일 적용")
    print("=" * 70)

    base_submission = load_csv_with_fallback("submission_sample.csv")
    local_idw = load_csv_with_fallback("4_local_IDW_submission.csv")
    regression_idw = load_csv_with_fallback("5_2_regression_IDW_submission.csv")

    if len(local_idw) != len(regression_idw):
        raise ValueError("두 제출물의 행 수가 다릅니다. 먼저 동일한 설정으로 다시 생성하세요.")

    if "time" in local_idw.columns and "time" in regression_idw.columns:
        if not local_idw["time"].equals(regression_idw["time"]) or not local_idw["pv_id"].equals(
            regression_idw["pv_id"]
        ):
            raise ValueError("time/pv_id 정렬이 맞지 않습니다. 상위 스크립트를 동일 조건으로 재생성하세요.")

    weight_sum = WEIGHT_LOCAL_IDW + WEIGHT_REG_IDW
    blended = (
        local_idw["nins"].values * WEIGHT_LOCAL_IDW + regression_idw["nins"].values * WEIGHT_REG_IDW
    ) / weight_sum

    scaled = np.clip(blended * SCALE_FACTOR, 0, None)  # 음수 제거

    final_submission = base_submission.copy()
    final_submission["nins"] = scaled
    final_submission.to_csv(OUTPUT_FILE, index=False)

    print(f"- local IDW mean: {local_idw['nins'].mean():.4f}")
    print(f"- regression IDW mean: {regression_idw['nins'].mean():.4f}")
    print(f"- blended mean (pre-scale): {blended.mean():.4f}")
    print(f"- blended mean (scaled): {final_submission['nins'].mean():.4f}")
    print(f"? Saved final ensemble submission -> {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
