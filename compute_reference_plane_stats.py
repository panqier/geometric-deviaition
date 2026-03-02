from pathlib import Path
import numpy as np
import pandas as pd

CRACK_CSV = "crack_points_xyz.csv"
LEFT_CSV = "left_points_xyz.csv"
RIGHT_CSV = "right_points_xyz.csv"
NONCRACK_CSV = "noncrack_points_xyz.csv"


def load_xyz(csv_path):
    df = pd.read_csv(csv_path)
    pts = df[["X", "Y", "Z"]].to_numpy(dtype=float)
    return df, pts


def fit_plane_svd(points):
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    return centroid, normal


def distances_to_plane(points, centroid, normal):
    centered = points - centroid
    signed_dist = centered @ normal
    abs_dist = np.abs(signed_dist)
    return signed_dist, abs_dist


def summarize(abs_dist):
    return {
        "n_points": len(abs_dist),
        "mean_abs_dist": float(np.mean(abs_dist)),
        "rms_dist": float(np.sqrt(np.mean(abs_dist ** 2))),
        "p95_dist": float(np.percentile(abs_dist, 95)),
        "max_abs_dist": float(np.max(abs_dist)),
    }


def analyze_against_reference(name, csv_path, ref_centroid, ref_normal):
    df, pts = load_xyz(csv_path)
    signed_dist, abs_dist = distances_to_plane(pts, ref_centroid, ref_normal)
    stats = summarize(abs_dist)

    out_df = df.copy()
    out_df["signed_dist_to_ref_plane"] = signed_dist
    out_df["abs_dist_to_ref_plane"] = abs_dist

    out_csv = f"{Path(csv_path).stem}_with_ref_plane_dist.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"\n=== {name} (against NON-CRACK reference plane) ===")
    print(f"Points: {stats['n_points']}")
    print(f"Mean abs distance: {stats['mean_abs_dist']:.6f}")
    print(f"RMS distance:      {stats['rms_dist']:.6f}")
    print(f"P95 distance:      {stats['p95_dist']:.6f}")
    print(f"Max abs distance:  {stats['max_abs_dist']:.6f}")
    print(f"Saved: {out_csv}")

    return {"group": name, **stats}


def main():
    # 1) 用 non-crack 拟合参考平面
    non_df, non_pts = load_xyz(NONCRACK_CSV)
    ref_centroid, ref_normal = fit_plane_svd(non_pts)

    print("Reference plane fitted from NON-CRACK points")
    print("Centroid:", ref_centroid)
    print("Normal:", ref_normal)

    # 2) 所有组都对同一个参考平面计算距离
    results = []
    results.append(analyze_against_reference("crack", CRACK_CSV, ref_centroid, ref_normal))
    results.append(analyze_against_reference("left_noncrack", LEFT_CSV, ref_centroid, ref_normal))
    results.append(analyze_against_reference("right_noncrack", RIGHT_CSV, ref_centroid, ref_normal))
    results.append(analyze_against_reference("noncrack_all", NONCRACK_CSV, ref_centroid, ref_normal))

    summary_df = pd.DataFrame(results)
    summary_df.to_csv("reference_plane_summary.csv", index=False)
    print("\nSaved summary: reference_plane_summary.csv")


if __name__ == "__main__":
    main()