import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import stats

K = 20  # 邻域大小：可试 10, 20, 30

def load_pts(path):
    df = pd.read_csv(path)
    pts = df[["X","Y","Z"]].to_numpy(dtype=float)
    return df, pts

def local_plane_residuals(points, k=K):
    """
    对每个点：kNN -> 拟合局部平面 -> 返回该点到局部平面的绝对距离
    """
    tree = cKDTree(points)
    # k+1 是因为最近的是自己
    dists, idx = tree.query(points, k=k+1)
    residuals = np.zeros(len(points), dtype=float)

    for i in range(len(points)):
        nb = points[idx[i, 1:]]  # 去掉自己
        c = nb.mean(axis=0)
        X = nb - c
        # SVD 拟合局部平面
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        n = vh[-1]
        n = n / np.linalg.norm(n)
        # 当前点到该局部平面的距离
        residuals[i] = abs((points[i] - c) @ n)

    return residuals

def summarize(x):
    return {
        "n": len(x),
        "mean": float(np.mean(x)),
        "rms": float(np.sqrt(np.mean(x**2))),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
        "std": float(np.std(x, ddof=1)),
    }

def main():
    groups = {
        "crack": "crack_points_xyz.csv",
        "left": "left_points_xyz.csv",
        "right": "right_points_xyz.csv",
        "noncrack": "noncrack_points_xyz.csv",
    }

    res_map = {}
    summary_rows = []

    for name, path in groups.items():
        df, pts = load_pts(path)
        r = local_plane_residuals(pts, k=K)
        df["local_roughness"] = r
        out = path.replace(".csv", f"_roughness_k{K}.csv")
        df.to_csv(out, index=False)

        s = summarize(r)
        s["group"] = name
        summary_rows.append(s)
        res_map[name] = r
        print(name, s)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"local_roughness_summary_k{K}.csv", index=False)

    # 统计检验：Welch + effect size
    def cohens_d(a,b):
        a=np.asarray(a); b=np.asarray(b)
        na, nb=len(a), len(b)
        va, vb=np.var(a, ddof=1), np.var(b, ddof=1)
        pooled=np.sqrt(((na-1)*va + (nb-1)*vb)/(na+nb-2))
        return 0.0 if pooled==0 else (np.mean(a)-np.mean(b))/pooled

    comparisons = [
        ("crack vs noncrack", "crack", "noncrack"),
        ("crack vs left", "crack", "left"),
        ("crack vs right", "crack", "right"),
        ("left vs right", "left", "right"),
    ]

    rows=[]
    for title,a,b in comparisons:
        A=res_map[a]; B=res_map[b]
        t,p = stats.ttest_ind(A,B,equal_var=False)
        d = cohens_d(A,B)
        rows.append({
            "comparison": title,
            "mean_a": float(np.mean(A)), "mean_b": float(np.mean(B)),
            "t": float(t), "p": float(p), "d": float(d)
        })
    test_df=pd.DataFrame(rows)
    test_df.to_csv(f"local_roughness_tests_k{K}.csv", index=False)
    print(test_df)

if __name__ == "__main__":
    main()