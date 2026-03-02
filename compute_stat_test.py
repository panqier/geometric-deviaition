import numpy as np
import pandas as pd
from scipy import stats

CRACK_CSV = "crack_points_xyz_with_ref_plane_dist.csv"
NONCRACK_CSV = "noncrack_points_xyz_with_ref_plane_dist.csv"


def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nx = len(x)
    ny = len(y)

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    pooled_std = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def main():
    crack_df = pd.read_csv(CRACK_CSV)
    non_df = pd.read_csv(NONCRACK_CSV)

    crack = crack_df["abs_dist_to_ref_plane"].to_numpy(dtype=float)
    noncrack = non_df["abs_dist_to_ref_plane"].to_numpy(dtype=float)

    # Welch's t-test
    t_stat, p_val = stats.ttest_ind(crack, noncrack, equal_var=False)

    # Effect size
    d = cohens_d(crack, noncrack)

    print("=== Statistical Comparison: crack vs non-crack ===")
    print(f"Crack n = {len(crack)}, mean = {np.mean(crack):.6f}, std = {np.std(crack, ddof=1):.6f}")
    print(f"Non-crack n = {len(noncrack)}, mean = {np.mean(noncrack):.6f}, std = {np.std(noncrack, ddof=1):.6f}")
    print(f"Welch t-statistic = {t_stat:.6f}")
    print(f"p-value = {p_val:.6f}")
    print(f"Cohen's d = {d:.6f}")

    result_df = pd.DataFrame([{
        "crack_n": len(crack),
        "noncrack_n": len(noncrack),
        "crack_mean": np.mean(crack),
        "noncrack_mean": np.mean(noncrack),
        "t_statistic": t_stat,
        "p_value": p_val,
        "cohens_d": d,
    }])
    result_df.to_csv("stat_test_summary.csv", index=False)
    print("Saved: stat_test_summary.csv")


if __name__ == "__main__":
    main()