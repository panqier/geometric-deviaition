from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

IMAGES_TXT = "sparse/0/images.txt"
POINTS3D_TXT = "sparse/0/points3D.txt"
TRAIN_MASKS_DIR = Path("train_masks")

# 颜色（你的mask用的是纯色）
WHITE = np.array([255, 255, 255])  # crack
GREEN = np.array([0, 255, 0])      # left non-crack
BLUE  = np.array([0, 0, 255])      # right non-crack
COLOR_TOL = 10

def color_match(pixel, target, tol=COLOR_TOL):
    return np.all(np.abs(pixel.astype(int) - target.astype(int)) <= tol)

def parse_colmap_images_txt(path):
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

        records.append({"image_id": image_id, "image_name": image_name, "points2d": points2d})
        i += 2

    return records

def load_points3d_xyz(path):
    pts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pid = int(parts[0])
            x, y, z = map(float, parts[1:4])
            pts[pid] = (x, y, z)
    return pts

def main():
    records = parse_colmap_images_txt(IMAGES_TXT)
    points3d_xyz = load_points3d_xyz(POINTS3D_TXT)

    crack_ids, left_ids, right_ids = set(), set(), set()
    used_images = 0

    # mask文件名就是 IMG_0525.PNG 这种（注意扩展名可能是 .PNG）
    # 统一用 stem 匹配：IMG_0525
    mask_map = {p.stem: p for p in TRAIN_MASKS_DIR.glob("*.PNG")}
    mask_map.update({p.stem: p for p in TRAIN_MASKS_DIR.glob("*.png")})

    for rec in records:
        stem = Path(rec["image_name"]).stem  # IMG_0525
        if stem not in mask_map:
            continue

        mask_path = mask_map[stem]
        mask = np.array(Image.open(mask_path).convert("RGB"))
        h, w = mask.shape[:2]

        used_images += 1
        print(f"Processing: {rec['image_name']}  <->  {mask_path.name}")

        for x, y, p3d_id in rec["points2d"]:
            if p3d_id == -1:
                continue
            xi, yi = int(round(x)), int(round(y))
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                continue

            px = mask[yi, xi]
            if color_match(px, WHITE):
                crack_ids.add(p3d_id)
            elif color_match(px, GREEN):
                left_ids.add(p3d_id)
            elif color_match(px, BLUE):
                right_ids.add(p3d_id)

    noncrack_ids = left_ids.union(right_ids)

    print("\n=== Summary ===")
    print(f"Used images: {used_images}")
    print(f"Crack 3D points: {len(crack_ids)}")
    print(f"Left 3D points: {len(left_ids)}")
    print(f"Right 3D points: {len(right_ids)}")
    print(f"Non-crack total 3D points: {len(noncrack_ids)}")

    def export_xyz(ids_set, out_csv):
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("POINT3D_ID,X,Y,Z\n")
            for pid in sorted(ids_set):
                if pid in points3d_xyz:
                    x, y, z = points3d_xyz[pid]
                    f.write(f"{pid},{x},{y},{z}\n")

    export_xyz(crack_ids, "crack_points_xyz.csv")
    export_xyz(left_ids, "left_points_xyz.csv")
    export_xyz(right_ids, "right_points_xyz.csv")
    export_xyz(noncrack_ids, "noncrack_points_xyz.csv")

if __name__ == "__main__":
    main()