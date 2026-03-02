from pathlib import Path
from PIL import Image
import numpy as np

# ====== 路径 ======
IMAGES_TXT = "sparse/0/images.txt"
POINTS3D_TXT = "sparse/0/points3D.txt"
TRAIN_IMAGES_DIR = Path("train_images")
TRAIN_MASKS_DIR = Path("train_masks")

# ====== 颜色定义（必须和你的mask一致）======
WHITE = np.array([255, 255, 255])  # crack
GREEN = np.array([0, 255, 0])      # left non-crack
BLUE  = np.array([0, 0, 255])      # right non-crack

# 如果颜色有轻微偏差，可放宽阈值
COLOR_TOL = 10

def color_match(pixel, target, tol=COLOR_TOL):
    return np.all(np.abs(pixel.astype(int) - target.astype(int)) <= tol)

def parse_colmap_images_txt(path):
    """
    读取 COLMAP text model 的 images.txt
    返回：
      records = [
        {
          "image_id": int,
          "image_name": str,
          "points2d": [(x, y, point3d_id), ...]
        },
        ...
      ]
    """
    records = []
    lines = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)

    i = 0
    while i < len(lines):
        header = lines[i].split()
        pts_line = lines[i + 1].split()

        image_id = int(header[0])
        image_name = header[-1]

        points2d = []
        for j in range(0, len(pts_line), 3):
            x = float(pts_line[j])
            y = float(pts_line[j + 1])
            p3d_id = int(float(pts_line[j + 2]))
            points2d.append((x, y, p3d_id))

        records.append({
            "image_id": image_id,
            "image_name": image_name,
            "points2d": points2d
        })

        i += 2

    return records

def load_points3d_xyz(path):
    """
    读取 points3D.txt
    返回 dict:
      point_id -> (x, y, z)
    """
    pts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            point_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            pts[point_id] = (x, y, z)
    return pts

def build_mask_name_from_image_name(image_name):
    """
    image001.jpeg -> mask001.png
    按你当前命名规则映射
    """
    stem = Path(image_name).stem  # image001
    suffix_num = stem.replace("image", "")
    return f"mask{suffix_num}.png"

def main():
    records = parse_colmap_images_txt(IMAGES_TXT)
    points3d_xyz = load_points3d_xyz(POINTS3D_TXT)

    crack_ids = set()
    left_ids = set()
    right_ids = set()

    used_images = 0

    for rec in records:
        image_name = rec["image_name"]
        mask_name = Path(image_name).with_suffix(".png").name
        mask_path = TRAIN_MASKS_DIR / mask_name

        if not mask_path.exists():
            continue

        mask = np.array(Image.open(mask_path).convert("RGB"))
        h, w = mask.shape[:2]

        used_images += 1
        print(f"Processing: {image_name} <-> {mask_name}")

        for x, y, p3d_id in rec["points2d"]:
            if p3d_id == -1:
                continue

            xi = int(round(x))
            yi = int(round(y))

            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                continue

            pixel = mask[yi, xi]

            if color_match(pixel, WHITE):
                crack_ids.add(p3d_id)
            elif color_match(pixel, GREEN):
                left_ids.add(p3d_id)
            elif color_match(pixel, BLUE):
                right_ids.add(p3d_id)

    noncrack_ids = left_ids.union(right_ids)

    print("\n=== Summary ===")
    print(f"Used images: {used_images}")
    print(f"Crack 3D points: {len(crack_ids)}")
    print(f"Left 3D points: {len(left_ids)}")
    print(f"Right 3D points: {len(right_ids)}")
    print(f"Non-crack total 3D points: {len(noncrack_ids)}")

    # 导出点ID
    with open("crack_point_ids.txt", "w", encoding="utf-8") as f:
        for pid in sorted(crack_ids):
            f.write(f"{pid}\n")

    with open("left_point_ids.txt", "w", encoding="utf-8") as f:
        for pid in sorted(left_ids):
            f.write(f"{pid}\n")

    with open("right_point_ids.txt", "w", encoding="utf-8") as f:
        for pid in sorted(right_ids):
            f.write(f"{pid}\n")

    with open("noncrack_point_ids.txt", "w", encoding="utf-8") as f:
        for pid in sorted(noncrack_ids):
            f.write(f"{pid}\n")

    # 导出XYZ坐标
    def export_xyz(ids_set, out_path):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("POINT3D_ID,X,Y,Z\n")
            for pid in sorted(ids_set):
                if pid in points3d_xyz:
                    x, y, z = points3d_xyz[pid]
                    f.write(f"{pid},{x},{y},{z}\n")

    export_xyz(crack_ids, "crack_points_xyz.csv")
    export_xyz(left_ids, "left_points_xyz.csv")
    export_xyz(right_ids, "right_points_xyz.csv")
    export_xyz(noncrack_ids, "noncrack_points_xyz.csv")

    print("\nExported:")
    print("- crack_points_xyz.csv")
    print("- left_points_xyz.csv")
    print("- right_points_xyz.csv")
    print("- noncrack_points_xyz.csv")

if __name__ == "__main__":
    main()