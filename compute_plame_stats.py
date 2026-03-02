from pathlib import Path
import numpy as np
import pandas as pd

# ====== 输入文件 ======
CRACK_CSV = "crack_points_xyz.csv"
LEFT_CSV = "left_points_xyz.csv"
RIGHT_CSV = "right_points_xyz.csv"
NONCRACK_CSV = "noncrack_points_xyz.csv"


def load_xyz(csv_path):
    df = pd.read_csv(csv_path)
    pts = df[["X", "Y", "Z"]].to_numpy(dtype=float)
    return df, pts


def fit_plane_svd(points):
    """
    对点集拟合最佳平面:
    返回:
      centroid: 平面中心点
      normal: 平面法向
      signed_dist: 每个点到平面的有符号距离
      abs_dist: 绝对距离
    """
    centroid = points.mean(axis=0)
    centered = points - centroid

    # SVD: 最小奇异值对应方向就是法向
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)

    signed_dist = centered @ normal
    abs_dist = np.abs(signed_dist)

    return centroid, normal, signed_dist, abs_dist


def summarize_distances(abs_dist):
    rms = np.sqrt(np.mean(abs_dist ** 2))
    mean_abs = np.mean(abs_dist)
    p95 = np.percentile(abs_dist, 95)
    max_abs = np.max(abs_dist)

    return {
        "n_points": len(abs_dist),
        "mean_abs_dist": mean_abs,
        "rms_dist": rms,
        "p95_dist": p95,
        "max_abs_dist": max_abs,
    }


def analyze_group(name, csv_path):
    df, pts = load_xyz(csv_path)

    if len(pts) < 3:
        raise ValueError(f"{name}: not enough points to fit a plane (need at least 3).")

    centroid, normal, signed_dist, abs_dist = fit_plane_svd(pts)
    stats = summarize_distances(abs_dist)

    out_df = df.copy()
    out_df["signed_dist_to_plane"] = signed_dist
    out_df["abs_dist_to_plane"] = abs_dist

    out_csv = f"{Path(csv_path).stem}_with_plane_dist.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"\n=== {name} ===")
    print(f"Points: {stats['n_points']}")
    print(f"Mean abs distance: {stats['mean_abs_dist']:.6f}")
    print(f"RMS distance:      {stats['rms_dist']:.6f}")
    print(f"P95 distance:      {stats['p95_dist']:.6f}")
    print(f"Max abs distance:  {stats['max_abs_dist']:.6f}")
    print(f"Saved: {out_csv}")

    return {
        "group": name,
        **stats
    }


def main():
    results = []

    results.append(analyze_group("crack", CRACK_CSV))
    results.append(analyze_group("left_noncrack", LEFT_CSV))
    results.append(analyze_group("right_noncrack", RIGHT_CSV))
    results.append(analyze_group("noncrack_all", NONCRACK_CSV))

    summary_df = pd.DataFrame(results)
    summary_df.to_csv("plane_fit_summary.csv", index=False)

    print("\nSaved summary: plane_fit_summary.csv")


if __name__ == "__main__":
    main()