# UA-DETRAC validation: Quality Robustness Verification Using Real CCTV Footage

# (1) UA-DETRAC dataset Preprocessing
# "Metric of Actual Measurements" Extreme Value Standardization High/Low Validation Set
# ==========================================================================================
import os
import re
import csv
import glob
import shutil
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# directory path setup
DATASET_ROOT = r"/home/user/UA-DETRAC"           # downloaded UA-DETRAC dataset folder
OUTPUT_DIR = r"/home/user/validation_set"    

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

SCAN_STRIDE = 5              
MAX_SCAN_PER_SEQUENCE = 300  
QUALITY_PERCENTILE = 25     
N_PER_GROUP = 500            
MAX_PER_SEQUENCE = 80        
RANDOM_SEED = 42

def find_all_sequence_images(root):
    all_files = glob.glob(os.path.join(root, "**", "*"), recursive=True)
    by_seq = defaultdict(list)
    for f in all_files:
        if not f.lower().endswith(IMAGE_EXTS) or not os.path.isfile(f):
            continue
        seq_name = os.path.basename(os.path.dirname(f))
        by_seq[seq_name].append(f)
    for seq in by_seq:
        by_seq[seq].sort()
    return by_seq

def compute_image_quality_metrics(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    file_size_bytes = os.path.getsize(path)
    bpp = (file_size_bytes * 8.0) / (w * h) if w * h > 0 else float("nan")
    return {"sharpness": sharpness, "brightness": brightness, "bpp": bpp}

def scan_and_measure(by_seq, stride, max_scan_per_seq):
    print("=" * 78)
    print(f"sequence {len(by_seq)}ea image quality indicator measurement, Up to {max_scan_per_seq} frames per sequence.")
    print("=" * 78)
    rows = []
    n_failed = 0
    for i, (seq, files) in enumerate(sorted(by_seq.items()), start=1):
        sampled = files[::stride][:max_scan_per_seq]
        for path in sampled:
            m = compute_image_quality_metrics(path)
            if m is None:
                n_failed += 1
                continue
            rows.append({"sequence": seq, "path": path, "filename": os.path.basename(path), **m})
        if i % 20 == 0 or i == len(by_seq):
            print(f"  processing: {i}/{len(by_seq)} sequence, Cumulative measurement {len(rows)}ea")
    if n_failed:
        print(f"  [warning] read failure {n_failed}ea exception")
    print(f"  Total measurement completed: {len(rows)}ea (sequence {len(by_seq)}ea)")
    return rows

def add_quality_score(rows):
    sharp = np.array([r["sharpness"] for r in rows], dtype=float)
    bright = np.array([r["brightness"] for r in rows], dtype=float)
    bpp = np.array([r["bpp"] for r in rows], dtype=float)

    def zscore(x):
        sd = x.std()
        return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)

    score = zscore(sharp) + zscore(bright) + zscore(bpp)
    for r, s in zip(rows, score):
        r["quality_score"] = float(s)
    return rows

def build_candidate_pools(rows, percentile):
    scores = np.array([r["quality_score"] for r in rows])
    high_thresh = np.percentile(scores, 100 - percentile)
    low_thresh = np.percentile(scores, percentile)
    high_pool = [r for r in rows if r["quality_score"] >= high_thresh]
    low_pool = [r for r in rows if r["quality_score"] <= low_thresh]
    print("\n" + "=" * 78)
    print(f"Candidate pool composition (Upper/Lower {percentile}%)")
    print("=" * 78)
    print(f"  High threshold: quality_score >= {high_thresh:.3f}  ->  candidate {len(high_pool)}ea "
          f"(sequence {len({r['sequence'] for r in high_pool})}ea)")
    print(f"  Low  threshold: quality_score <= {low_thresh:.3f}  ->  candidate {len(low_pool)}ea "
          f"(sequence {len({r['sequence'] for r in low_pool})}ea)")
    return {"High": high_pool, "Low": low_pool}

def capped_sample(pool, n_target, max_per_seq, rng):
    by_seq = defaultdict(list)
    for r in pool:
        by_seq[r["sequence"]].append(r)

    capped_pool = []
    for seq, items in by_seq.items():
        items = items[:]
        rng.shuffle(items)
        capped_pool.extend(items if max_per_seq is None else items[:max_per_seq])

    if len(capped_pool) >= n_target:
        return rng.sample(capped_pool, n_target)

    print(f"  When applying a per-sequence cap ({max_per_seq}), use {len(capped_pool)}If the length is insufficient to reach the target ({n_target}) -> replenish without an upper limit.")
    chosen = list(capped_pool)
    remaining = [r for r in pool if r not in chosen]
    need = n_target - len(chosen)
    if len(remaining) < need:
        print(f"  All candidates ({len(pool)}The length is also less than (n_target). As much as possible (({len(pool)}))ea only use.")
        return pool
    return chosen + rng.sample(remaining, need)

def copy_and_write_manifest(pools_selected, output_dir):
    print("\n" + "=" * 78)
    print("Copy and save the manifest")
    print("=" * 78)
    manifest_rows = []
    for group, selected in pools_selected.items():
        out_dir = os.path.join(output_dir, group)
        os.makedirs(out_dir, exist_ok=True)
        n_copied, n_skipped = 0, 0
        for r in selected:
            dst_name = f"{r['sequence']}__{r['filename']}"
            dst = os.path.join(out_dir, dst_name)
            if not os.path.exists(dst):
                shutil.copy2(r["path"], dst)
                n_copied += 1
            else:
                n_skipped += 1
            manifest_rows.append({
                "quality_group": group, "sequence": r["sequence"], "filename": r["filename"],
                "quality_score": r["quality_score"], "sharpness": r["sharpness"],
                "brightness": r["brightness"], "bpp": r["bpp"],
                "src_path": r["path"], "dst_path": dst,
            })
        print(f"  {group}: {len(selected)}ea (New copy {n_copied}, Maintain existing {n_skipped})")

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "validation_set_v2_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "quality_group", "sequence", "filename", "quality_score",
            "sharpness", "brightness", "bpp", "src_path", "dst_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\n  manifest saved: {manifest_path} (Total {len(manifest_rows)}row)")

    for group in pools_selected:
        vals = [r["quality_score"] for r in manifest_rows if r["quality_group"] == group]
        if vals:
            print(f"  {group} quality_score: mean={np.mean(vals):.3f}, "
                  f"min={np.min(vals):.3f}, max={np.max(vals):.3f}")
    return manifest_path

def main():
    assert os.path.isdir(DATASET_ROOT), f"DATASET_ROOT cannot be found: {DATASET_ROOT}"
    by_seq = find_all_sequence_images(DATASET_ROOT)
    assert by_seq, f"No image files were found under {DATASET_ROOT}."
    print(f"founded sequence: {len(by_seq)}ea, Total image (before scan): "
          f"{sum(len(v) for v in by_seq.values())}ea")

    rows = scan_and_measure(by_seq, SCAN_STRIDE, MAX_SCAN_PER_SEQUENCE)
    assert rows, "No image has been measured for quality indicators. Check the SCAN_STRIDE path."
    rows = add_quality_score(rows)

    pools = build_candidate_pools(rows, QUALITY_PERCENTILE)
    rng = random.Random(RANDOM_SEED)
    selected = {
        group: capped_sample(pool, N_PER_GROUP, MAX_PER_SEQUENCE, rng)
        for group, pool in pools.items()
    }

    copy_and_write_manifest(selected, OUTPUT_DIR)
    print("\nCompleted!")


if __name__ == "__main__":
    main()



# (2) Run the Validation
# ========================================================================================================
import subprocess
import sys

def ensure_env():
    pkgs = {
        "ultralytics": "ultralytics",
        "PIL": "pillow",
        "cv2": "opencv-python-headless",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
    }
    for mod, pip_name in pkgs.items():
        try:
            __import__(mod)
        except ImportError:
            print(f"[env] installing {pip_name} ...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=True)

ensure_env()

import os
import re
import csv
import glob
import shutil
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

VALID_ROOT = "/content/drive/MyDrive/UA-DETRAC"       # Validation Set ("Parent" directory of the High/Low image folder)
XML_ROOT = "/content/drive/MyDrive/UA-DETRAC/xml"     # Original UA-DETRAC XML comments folder
WEIGHTS = {
    "Standard": "/content/drive/MyDrive/CCTV_MultiRes_Experiments/experiments/exp1_standard_a100_20260709_041147/weights/best.pt",            # best weights for the Standard model
    "MultiQuality": "/content/drive/MyDrive/CCTV_MultiRes_Experiments/experiments/exp2_multiquality_a100_20260709_053822/weights/best.pt",    # best weights for the Multi-Quality model
}

OUTPUT_DIR = "/content/drive/MyDrive/UA-DETRAC/r3_11_outputs"
LOCAL_DATASET_DIR = "/content/r3_11_dataset2"

VEHICLE_TYPE_TO_CLASS_NAME = {
    "car": "car",
    "van": "van",
    "bus": "bus", 
}

IMGSZ = 640     
CONF_THRES = 0.001
IOU_THRES_NMS = 0.6
GROUPS = ("High", "Low")
RANDOM_SEED = 42


from PIL import Image

FNAME_RE = re.compile(r"(\d+)\.\w+$")  

def list_group_images(valid_root, group):
    d = os.path.join(valid_root, group)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"cannot find the folder: {d}")
    exts = (".jpg", ".jpeg", ".png")
    return sorted(f for f in os.listdir(d) if f.lower().endswith(exts))

def parse_filename(fn):
    if "__" not in fn:
        return None, None
    seq, orig = fn.split("__", 1)
    m = FNAME_RE.search(orig)
    if not m:
        return seq, None
    return seq, int(m.group(1))

def find_sequence_xml(xml_root, seq_name):
    cands = glob.glob(os.path.join(xml_root, "**", f"{seq_name}*.xml"), recursive=True)
    return cands[0] if cands else None

def parse_sequence_targets(xml_path):
    frame_targets = {}
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        print(f"  [warning] XML parsing failure: {xml_path} ({e})")
        return frame_targets
    root = tree.getroot()
    for frame_el in root.findall("frame"):
        try:
            fnum = int(frame_el.attrib["num"])
        except (KeyError, ValueError):
            continue
        targets = []
        tlist = frame_el.find("target_list")
        if tlist is None:
            frame_targets[fnum] = targets
            continue
        for tgt in tlist.findall("target"):
            box_el = tgt.find("box")
            attr_el = tgt.find("attribute")
            if box_el is None:
                continue
            vtype = (attr_el.attrib.get("vehicle_type") if attr_el is not None else None) or "unknown"
            try:
                l = float(box_el.attrib["left"])
                t = float(box_el.attrib["top"])
                w = float(box_el.attrib["width"])
                h = float(box_el.attrib["height"])
            except (KeyError, ValueError):
                continue
            targets.append((vtype.strip().lower(), l, t, w, h))
        frame_targets[fnum] = targets
    return frame_targets

def box_to_yolo_line(class_id, l, t, w, h, img_w, img_h):
    cx = (l + w / 2.0) / img_w
    cy = (t + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    cx, cy, nw, nh = (max(0.0, min(1.0, v)) for v in (cx, cy, nw, nh))
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

def build_gt_and_local_dataset(valid_root, xml_root, local_root,
                                class_name_to_id, vehicle_type_to_class_name):
    stats = {
        "images_total": 0, "images_used": 0, "images_excluded_no_xml": 0,
        "images_excluded_bad_filename": 0, "images_excluded_frame_not_found": 0,
        "boxes_written": 0, "boxes_excluded_unmapped_type": 0,
        "excluded_sequences": set(),
    }
    seq_xml_cache = {}       # seq_name -> xml_path or None
    seq_targets_cache = {}   # seq_name -> {frame_num: [...]}

    for group in GROUPS:
        img_dir = os.path.join(valid_root, group)
        out_img_dir = os.path.join(local_root, "images", group)
        out_lbl_dir = os.path.join(local_root, "labels", group)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        filenames = list_group_images(valid_root, group)
        print(f"\n[{group}] original image {len(filenames)}ea preprocessing ...")

        for fn in filenames:
            stats["images_total"] += 1
            seq, frame_num = parse_filename(fn)
            if seq is None or frame_num is None:
                stats["images_excluded_bad_filename"] += 1
                continue

            if seq not in seq_xml_cache:
                xml_path = find_sequence_xml(xml_root, seq)
                seq_xml_cache[seq] = xml_path
                seq_targets_cache[seq] = parse_sequence_targets(xml_path) if xml_path else {}

            xml_path = seq_xml_cache[seq]
            if xml_path is None:
                stats["images_excluded_no_xml"] += 1
                stats["excluded_sequences"].add(seq)
                continue

            targets = seq_targets_cache[seq].get(frame_num)
            if targets is None:
                stats["images_excluded_frame_not_found"] += 1
                continue

            src_img_path = os.path.join(img_dir, fn)
            with Image.open(src_img_path) as im:
                img_w, img_h = im.size

            lines = []
            for vtype, l, t, w, h in targets:
                cls_name = vehicle_type_to_class_name.get(vtype)
                if cls_name is None or cls_name not in class_name_to_id:
                    stats["boxes_excluded_unmapped_type"] += 1
                    continue
                cls_id = class_name_to_id[cls_name]
                lines.append(box_to_yolo_line(cls_id, l, t, w, h, img_w, img_h))
                stats["boxes_written"] += 1

            stem = os.path.splitext(fn)[0]
            dst_img_path = os.path.join(out_img_dir, fn)
            if not os.path.exists(dst_img_path):
                shutil.copy2(src_img_path, dst_img_path)
            with open(os.path.join(out_lbl_dir, f"{stem}.txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

            stats["images_used"] += 1

    return stats

def print_conversion_report(stats):
    print("\n" + "=" * 78)
    print("[Summary]")
    print("=" * 78)
    print(f"  Total images: {stats['images_total']}")
    print(f"  used(label generation completed): {stats['images_used']}")
    print(f"  Exception - Sequence without XML: {stats['images_excluded_no_xml']}")
    print(f"  Exception - File name pattern mismatch: {stats['images_excluded_bad_filename']}")
    print(f"  Exception - No frame number available: {stats['images_excluded_frame_not_found']}")
    print(f"  Number of GT boxes created: {stats['boxes_written']}")
    print(f"  Number of unmapped (excluded) vehicle_type boxes: {stats['boxes_excluded_unmapped_type']}")
    if stats["excluded_sequences"]:
        print(f"  Sequence completely excluded because XML could not be found: {sorted(stats['excluded_sequences'])}")


from ultralytics import YOLO

def get_shared_class_names(weight_paths):
    names_per_model = {}
    for tag, path in weight_paths.items():
        m = YOLO(path)
        names_per_model[tag] = m.names  # {id: name}
    ref_tag = next(iter(names_per_model))
    ref_names = names_per_model[ref_tag]
    for tag, names in names_per_model.items():
        assert names == ref_names, (
            f"The class structure between models is different! {ref_tag}={ref_names} vs {tag}={names} "
        )
    class_name_to_id = {name: idx for idx, name in ref_names.items()}
    print(f"[info] Shared class system verified: {ref_names}")
    return class_name_to_id

def write_data_yaml(local_root, group, class_name_to_id, out_yaml_path):
    id_to_name = {v: k for k, v in class_name_to_id.items()}
    names_block = "\n".join(f"  {i}: {id_to_name[i]}" for i in sorted(id_to_name))
    content = (
        f"path: {local_root}\n"
        f"train: images/{group}\n"  # ultralytics requires 'train' key even for eval-only use;
        f"val: images/{group}\n"    # we only ever call model.val(split='val'), so this is unused.
        f"names:\n{names_block}\n"
    )
    with open(out_yaml_path, "w") as f:
        f.write(content)
    return out_yaml_path

def evaluate_all(weight_paths, local_root, class_name_to_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for group in GROUPS:
        yaml_path = os.path.join(output_dir, f"data_{group}.yaml")
        write_data_yaml(local_root, group, class_name_to_id, yaml_path)

        for tag, w_path in weight_paths.items():
            print(f"\n[평가] model={tag}, group={group}")
            model = YOLO(w_path)
            metrics = model.val(
                data=yaml_path, split="val", imgsz=IMGSZ,
                conf=CONF_THRES, iou=IOU_THRES_NMS, verbose=False,
            )
            rows.append({
                "model": tag,
                "quality_group": group,
                "mAP50": float(metrics.box.map50),
                "mAP50_95": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
            })
    return rows


# Aggregation of Results + Saving CSV/Graphs
import pandas as pd
import matplotlib.pyplot as plt

COLOR_STANDARD = "#2a78d6"       
COLOR_MULTIQUALITY = "#eb6834"   

def summarize_and_plot(rows, output_dir):
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "r3_11_real_cctv_validation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nresult CSV saved: {csv_path}")
    print(df.to_string(index=False))

    # High -> Low : Calculation of Decline Rate (%)
    pivot = df.pivot(index="model", columns="quality_group", values="mAP50")
    drop_rows = []
    for model in pivot.index:
        high = pivot.loc[model, "High"]
        low = pivot.loc[model, "Low"]
        drop_pct = (high - low) / high * 100.0 if high > 0 else float("nan")
        drop_rows.append({"model": model, "mAP50_High": high, "mAP50_Low": low, "drop_pct": drop_pct})
    drop_df = pd.DataFrame(drop_rows)
    drop_csv_path = os.path.join(output_dir, "r3_11_real_cctv_quality_drop_summary.csv")
    drop_df.to_csv(drop_csv_path, index=False)
    print(f"\n[High->Low decline rate summary] (The lower the latency, the more robust it is in real-world CCTV environments)")
    print(drop_df.to_string(index=False))
    print(f"summary CSV saved: {drop_csv_path}")

    # Figure: Model X Quality Group Bundle Bar (categorical color = model)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    groups = list(GROUPS)
    x = range(len(groups))
    width = 0.35
    colors = {"Standard": COLOR_STANDARD, "MultiQuality": COLOR_MULTIQUALITY}
    for i, model in enumerate(pivot.index):
        vals = [pivot.loc[model, g] for g in groups]
        offset = (i - (len(pivot.index) - 1) / 2) * width
        bars = ax.bar([xi + offset for xi in x], vals, width=width,
                      label=model, color=colors.get(model, "#999999"))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 0.02, f"{v:.3f}",
                    ha="center", va="bottom", color="white", fontweight="bold", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylabel("mAP50 (real-world UA-DETRAC)")
    ax.set_title("Real-World CCTV Validation: High vs Low Quality Group")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig_path = os.path.join(output_dir, "r3_11_real_cctv_mAP50_comparison.png")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"figure saved: {fig_path}")
    return df, drop_df


def main():
    class_name_to_id = get_shared_class_names(WEIGHTS)

    stats = build_gt_and_local_dataset(
        VALID_ROOT, XML_ROOT, LOCAL_DATASET_DIR,
        class_name_to_id, VEHICLE_TYPE_TO_CLASS_NAME,
    )
    print_conversion_report(stats)

    if stats["images_used"] == 0:
        raise RuntimeError(
            "0 images to use for evaluation. Check the XML_ROOT path and the naming convention for sample_uadetrac_validation_set.py."
        )

    rows = evaluate_all(WEIGHTS, LOCAL_DATASET_DIR, class_name_to_id, OUTPUT_DIR)
    summarize_and_plot(rows, OUTPUT_DIR)
    print("\nCompleted!")


if __name__ == "__main__":
    main()