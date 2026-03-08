import numpy as np
import pandas as pd
from scipy import stats

def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    return 0.0 if pooled == 0 else (np.mean(x) - np.mean(y)) / pooled

def load_abs(path, col):
    df = pd.read_csv(path)
    return df[col].to_numpy(dtype=float)

def test_pair(name, a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)  # Welch
    d = cohens_d(a, b)
    return {
        "comparison": name,
        "n_a": len(a), "mean_a": float(np.mean(a)), "std_a": float(np.std(a, ddof=1)),
        "n_b": len(b), "mean_b": float(np.mean(b)), "std_b": float(np.std(b, ddof=1)),
        "t_stat": float(t), "p_value": float(p), "cohens_d": float(d)
    }

def main():
    crack = load_abs("crack_points_xyz_with_ref_plane_dist.csv", "abs_dist_to_ref_plane")
    left = load_abs("left_points_xyz_with_ref_plane_dist.csv", "abs_dist_to_ref_plane")
    right = load_abs("right_points_xyz_with_ref_plane_dist.csv", "abs_dist_to_ref_plane")
    non = load_abs("noncrack_points_xyz_with_ref_plane_dist.csv", "abs_dist_to_ref_plane")

    results = []
    results.append(test_pair("crack vs noncrack_all", crack, non))
    results.append(test_pair("crack vs left_noncrack", crack, left))
    results.append(test_pair("crack vs right_noncrack", crack, right))
    results.append(test_pair("left vs right", left, right))

    out = pd.DataFrame(results)
    out.to_csv("stat_test_summary_v2.csv", index=False)
    print(out)

if __name__ == "__main__":
    main()