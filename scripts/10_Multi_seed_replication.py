# Multi-Seed experiment: Standard vs Multi-Quality vs B3 (Random-Quality-5x)
# (n = 3, random seed = 0, 42, 123)

import os
import sys
import json
import math
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yaml

# directory path setup
GDRIVE_ROOT = '/content/drive/MyDrive'
EXPERIMENT_PATH = os.path.join(GDRIVE_ROOT, 'MultiRes_Experiments')
VISDRONE_ROOT = os.path.join(GDRIVE_ROOT, 'VisDrone')            # original VisDrone dataset path
SRC_YOLO_DIR = os.path.join(EXPERIMENT_PATH, 'yolo_dataset')     # preprocessed dataset path by prior experiment

SMOKE_TEST = False          # True: test run for code check (epochs=1, data 2%)
SEEDS = [0, 42, 123]        # Random seeds (n=3)
METHODS = ['standard', 'multiquality', 'b3']
QUALITY_LEVELS = [20, 40, 60, 80, 100]
EXTRA_EVAL_LEVELS = []     
GPU_USD_PER_HOUR = 1.86     
EXISTING_RUNS = {}
CLS_REMAP = False

OUT_DIR = os.path.join('/content/drive/MyDrive/MultiRes_Experiments', 'multiseed' + ('_smoke' if SMOKE_TEST else ''))
RUNS_DIR = os.path.join(OUT_DIR, 'runs')            # Drive: weights/results.csv
RESULTS_DIR = os.path.join(OUT_DIR, 'results')      # Drive: JSON results by run
SUMMARY_DIR = os.path.join(OUT_DIR, 'summary')      # Drive: summary CSV/figure
LOCAL_ROOT = '/content/work'                        # Colab local SSD (I/O accelation)
LOCAL_YOLO = os.path.join(LOCAL_ROOT, 'yolo_dataset')

if SMOKE_TEST:
    SEEDS = SEEDS[:1]

CLASS_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van',
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
ALL_EVAL_LEVELS = sorted(set(QUALITY_LEVELS + EXTRA_EVAL_LEVELS))
BATCH_SIZE = 48

BASE_TRAIN_ARGS = dict(
    epochs=1 if SMOKE_TEST else 100,
    batch=BATCH_SIZE, imgsz=640, optimizer='SGD',
    lr0=0.06, momentum=0.9, weight_decay=0.0005,
    warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
    patience=50, workers=8,
    save_period=-1,      
    plots=False,         
    pretrained=True, deterministic=True, exist_ok=True, verbose=False,
)
if SMOKE_TEST:
    BASE_TRAIN_ARGS['fraction'] = 0.02

for d in (RUNS_DIR, RESULTS_DIR, SUMMARY_DIR, LOCAL_ROOT):
    os.makedirs(d, exist_ok=True)

def ensure_env():
    # checking for drive mount, packages, GPU
    try:
        from google.colab import drive  
        if not os.path.exists(GDRIVE_ROOT):
            drive.mount('/content/drive')
    except ImportError:
        pass
    try:
        import ultralytics  
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    import torch
    import ultralytics
    from ultralytics.cfg import DEFAULT_CFG_DICT
    if 'cls_remap' in DEFAULT_CFG_DICT:
        BASE_TRAIN_ARGS['cls_remap'] = CLS_REMAP
    assert torch.cuda.is_available(), 'GPU runtime (A100 recommend)'
    print(f'GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | '
          f'ultralytics {ultralytics.__version__} | cls_remap='
          f'{BASE_TRAIN_ARGS.get("cls_remap", "(n/a)")}')
    assert os.path.isdir(SRC_YOLO_DIR), f'There is no existing YOLO dataset: {SRC_YOLO_DIR}'
    return ultralytics.__version__


import cv2  
from tqdm.auto import tqdm 

class VisDroneParser:
    VALID_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    @staticmethod
    def parse_annotation(annotation_file):
        annotations = []
        if not os.path.exists(annotation_file):
            return annotations
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                x, y, w, h = map(int, parts[:4])
                cat = int(parts[5])
                if cat == 0 or w <= 0 or h <= 0:
                    continue
                if cat in VisDroneParser.VALID_CLASSES:
                    annotations.append({'bbox': [x, y, w, h], 'class': cat})
        return annotations

    @staticmethod
    def visdrone_to_yolo(annotations, img_w, img_h):
        out = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            cid = VisDroneParser.VALID_CLASSES.index(ann['class'])
            out.append({'class': cid, 'bbox': [
                max(0, min(1, (x + w / 2) / img_w)), max(0, min(1, (y + h / 2) / img_h)),
                max(0, min(1, w / img_w)), max(0, min(1, h / img_h))]})
        return out

class QualityDegrader:
    JPEG_QUALITY_MAP = {0.2: 20, 0.4: 40, 0.6: 60, 0.8: 80, 1.0: 100}

    @staticmethod
    def get_jpeg_quality(level):
        m = QualityDegrader.JPEG_QUALITY_MAP
        if level in m:
            return m[level]
        keys = sorted(m)
        if level <= keys[0]:
            return m[keys[0]]
        if level >= keys[-1]:
            return m[keys[-1]]
        for lo, hi in zip(keys[:-1], keys[1:]):
            if lo <= level <= hi:
                return int(round(m[lo] + (m[hi] - m[lo]) * (level - lo) / (hi - lo)))
        return int(np.clip(level * 100, 5, 95))

    @staticmethod
    def degrade_image(image, level):
        if level == 1.0:
            deg = image.copy()
        else:
            h, w = image.shape[:2]
            nh, nw = max(1, int(h * level)), max(1, int(w * level))
            down = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
            deg = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
        q = QualityDegrader.get_jpeg_quality(level)
        ok, buf = cv2.imencode('.jpg', deg, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else deg

def _n_workers():
    return max(2, min(16, os.cpu_count() or 4))

def _rsync(src, dst):
    os.makedirs(dst, exist_ok=True)
    if shutil.which('rsync'):
        subprocess.run(['rsync', '-a', src.rstrip('/') + '/', dst.rstrip('/') + '/'], check=True)
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)

def sync_dataset_to_local():
    subsets = ([('train', q) for q in QUALITY_LEVELS] + [('val', 100)]
               + [('test', q) for q in QUALITY_LEVELS])
    for split, q in tqdm(subsets, desc='Locally copy the dataset'):
        for kind in ('images', 'labels'):
            src = os.path.join(SRC_YOLO_DIR, kind, split, f'q{q}')
            dst = os.path.join(LOCAL_YOLO, kind, split, f'q{q}')
            if os.path.isdir(dst) and len(os.listdir(dst)) == len(os.listdir(src)):
                continue
            _rsync(src, dst)

def build_extra_test_levels():
    if not EXTRA_EVAL_LEVELS:
        return
    raw_dir = os.path.join(VISDRONE_ROOT, 'VisDrone2019-DET-test-dev', 'images')
    label_src = os.path.join(LOCAL_YOLO, 'labels', 'test', 'q100')
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(('.jpg', '.png')))
    for lv in EXTRA_EVAL_LEVELS:
        img_dst = os.path.join(LOCAL_YOLO, 'images', 'test', f'q{lv}')
        lbl_dst = os.path.join(LOCAL_YOLO, 'labels', 'test', f'q{lv}')
        if os.path.isdir(img_dst) and len(os.listdir(img_dst)) == len(os.listdir(label_src)):
            continue
        os.makedirs(img_dst, exist_ok=True)
        shutil.copytree(label_src, lbl_dst, dirs_exist_ok=True)
        level = lv / 100.0
        jq = QualityDegrader.get_jpeg_quality(level)

        def work(f):
            if not os.path.exists(os.path.join(label_src, os.path.splitext(f)[0] + '.txt')):
                return
            img = cv2.imread(os.path.join(raw_dir, f))
            out = QualityDegrader.degrade_image(img, level)
            cv2.imwrite(os.path.join(img_dst, f), out, [int(cv2.IMWRITE_JPEG_QUALITY), int(jq)])

        with ThreadPoolExecutor(_n_workers()) as ex:
            list(tqdm(ex.map(work, files), total=len(files), desc=f'additional experiment Q{lv}'))

def build_b3_dataset(seed, limit=None):
    root = os.path.join(LOCAL_ROOT, f'b3_seed{seed}')
    marker = os.path.join(root, '.complete')
    tag = 'random5x'
    if os.path.exists(marker):
        return os.path.join(root, 'images', 'train', tag)
    shutil.rmtree(root, ignore_errors=True)

    raw = os.path.join(VISDRONE_ROOT, 'VisDrone2019-DET-train')
    img_dir, ann_dir = os.path.join(raw, 'images'), os.path.join(raw, 'annotations')
    files = sorted(f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png')))
    if limit:
        files = files[:limit]

    rng = np.random.default_rng(seed)
    levels = rng.uniform(0.15, 1.0, size=(len(files), 5))     
    for k in range(5):
        os.makedirs(os.path.join(root, 'images', 'train', tag, f'copy{k}'), exist_ok=True)
        os.makedirs(os.path.join(root, 'labels', 'train', tag, f'copy{k}'), exist_ok=True)

    def work(i):
        f = files[i]
        stem = os.path.splitext(f)[0]
        anns = VisDroneParser.parse_annotation(os.path.join(ann_dir, stem + '.txt'))
        img = cv2.imread(os.path.join(img_dir, f))
        if img is None or not anns:
            return 0
        h, w = img.shape[:2]
        yolo = VisDroneParser.visdrone_to_yolo(anns, w, h)
        for k in range(5):
            lv = float(levels[i, k])
            jq = QualityDegrader.get_jpeg_quality(lv)
            out = QualityDegrader.degrade_image(img, lv)
            cv2.imwrite(os.path.join(root, 'images', 'train', tag, f'copy{k}', f), out,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(jq)])
            with open(os.path.join(root, 'labels', 'train', tag, f'copy{k}', stem + '.txt'), 'w') as fh:
                for a in yolo:
                    x, y, wn, hn = a['bbox']
                    fh.write(f"{a['class']} {x:.6f} {y:.6f} {wn:.6f} {hn:.6f}\n")
        return 1

    with ThreadPoolExecutor(_n_workers()) as ex:
        done = sum(tqdm(ex.map(work, range(len(files))), total=len(files),
                        desc=f'B3 data generation (seed={seed})'))
    np.save(os.path.join(OUT_DIR, f'b3_levels_seed{seed}.npy'), levels)   
    print(f'  B3(seed={seed}): {done}ea x 5 | quality mean={levels.mean():.3f} sd={levels.std():.3f}')
    Path(marker).touch()
    return os.path.join(root, 'images', 'train', tag)

def count_images(path):
    n = 0
    for _, _, fs in os.walk(path):
        n += sum(1 for f in fs if f.lower().endswith(('.jpg', '.png')))
    return n

def write_yaml(name, train):
    y = {'train': train, 'val': os.path.join(LOCAL_YOLO, 'images', 'val', 'q100'),
         'nc': len(CLASS_NAMES), 'names': CLASS_NAMES}
    p = os.path.join(LOCAL_ROOT, f'{name}.yaml')
    with open(p, 'w') as f:
        yaml.dump(y, f, default_flow_style=False)
    return p

def prepare_data(method, seed):
    if method == 'standard':
        train = os.path.join(LOCAL_YOLO, 'images', 'train', 'q100')
        return write_yaml('standard', train), count_images(train)
    if method == 'multiquality':
        paths = []
        for q in QUALITY_LEVELS:                                   
            d = os.path.join(LOCAL_YOLO, 'images', 'train', f'q{q}')
            paths += [os.path.abspath(os.path.join(d, f)) for f in sorted(os.listdir(d))
                      if f.endswith(('.jpg', '.png'))]
        lst = os.path.join(LOCAL_ROOT, 'multiquality_train.txt')
        with open(lst, 'w') as f:
            f.write('\n'.join(paths))
        return write_yaml('multiquality', lst), len(paths)
    if method == 'b3':
        train = build_b3_dataset(seed, limit=100 if SMOKE_TEST else None)
        return write_yaml(f'b3_seed{seed}', train), count_images(train)
    raise ValueError(method)

def train_run(method, seed, data_yaml):
    from ultralytics import YOLO
    name = f'{method}_seed{seed}'
    run_dir = Path(RUNS_DIR) / name
    done = run_dir / 'train_done.json'
    if done.exists():
        print(f'  [skip] training completed: {name}')
        return run_dir
    last = run_dir / 'weights' / 'last.pt'
    t0 = time.time()
    if last.exists():
        print(f'  [continue training] {last}')
        YOLO(str(last)).train(resume=True)
    else:
        args = dict(BASE_TRAIN_ARGS)
        args.update(data=data_yaml, project=RUNS_DIR, name=name, seed=seed)
        YOLO('yolov8m.pt').train(**args)
    done.write_text(json.dumps({'finished_at': datetime.now().isoformat(),
                                'wall_sec_this_session': time.time() - t0}))
    return run_dir

def evaluate_model(best_pt):
    from ultralytics import YOLO
    model = YOLO(str(best_pt))
    out = {}
    for lv in ALL_EVAL_LEVELS:
        yml = os.path.join(LOCAL_ROOT, f'eval_test_q{lv}.yaml')
        with open(yml, 'w') as f:
            yaml.dump({'train': os.path.join(LOCAL_YOLO, 'images', 'test', f'q{lv}'),
                       'val': os.path.join(LOCAL_YOLO, 'images', 'test', f'q{lv}'),
                       'test': os.path.join(LOCAL_YOLO, 'images', 'test', f'q{lv}'),
                       'nc': len(CLASS_NAMES), 'names': CLASS_NAMES}, f)
        m = model.val(data=yml, split='test', batch=16, imgsz=640, verbose=False, plots=False,
                      project=os.path.join(LOCAL_ROOT, 'val_runs'), name='tmp', exist_ok=True)
        out[lv] = {'mAP50': float(m.box.map50), 'mAP50-95': float(m.box.map),
                   'mAP75': float(m.box.map75),
                   'precision': float(m.box.p.mean()), 'recall': float(m.box.r.mean())}
        print(f'    Q{lv:>3}: mAP50={out[lv]["mAP50"]:.4f}  mAP50-95={out[lv]["mAP50-95"]:.4f}')
    return out

def read_train_stats(run_dir, n_train):
    df = pd.read_csv(Path(run_dir) / 'results.csv')
    df.columns = [c.strip() for c in df.columns]
    fit = 0.1 * df['metrics/mAP50(B)'] + 0.9 * df['metrics/mAP50-95(B)']
    bi = int(fit.values.argmax())
    epochs = int(len(df))
    t = df['time'].values.astype(float) if 'time' in df.columns else None
    if t is not None:
        d = np.diff(t, prepend=0.0)
        d[d < 0] = t[d < 0]                       
        total_sec, cum = float(d.sum()), np.cumsum(d)
        to_best_sec = float(cum[bi])
    else:
        total_sec = to_best_sec = float('nan')
    steps = math.ceil(n_train / BATCH_SIZE)
    total_hr = total_sec / 3600.0
    return {'n_train_images': int(n_train), 'epochs_trained': epochs,
            'best_epoch_fitness': int(df['epoch'].iloc[bi]) if 'epoch' in df.columns else bi + 1,
            'steps_per_epoch': steps, 'total_optimizer_updates': steps * epochs,
            'images_seen': int(n_train) * epochs,
            'total_time_hr': total_hr, 'time_to_best_hr': to_best_sec / 3600.0,
            'time_per_epoch_min': total_sec / 60.0 / epochs,
            'gpu_cost_usd': total_hr * GPU_USD_PER_HOUR}

def _auc(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    area = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
    return area / float(x[-1] - x[0])


def robustness_metrics(metrics, levels):
    lv = sorted(levels)
    x = np.array(lv) / 100.0
    out = {}
    for key, tag in (('mAP50', 'mAP50'), ('mAP50-95', 'mAP50-95')):
        y = np.array([metrics[q][key] for q in lv])
        ref = metrics[100][key]
        out[f'mean_{tag}'] = float(y.mean())
        out[f'sd_{tag}'] = float(y.std(ddof=0))
        out[f'score_{tag}'] = float(y.mean() / y.std(ddof=0)) if y.std(ddof=0) > 0 else float('nan')
        out[f'auc_{tag}'] = _auc(y, x)
        out[f'worst_{tag}'] = float(y.min())
        out[f'reldeg_Q20_{tag}'] = float(1.0 - metrics[min(lv)][key] / ref)
        out[f'reldeg_mean_{tag}'] = float(np.mean([1.0 - metrics[q][key] / ref for q in lv if q != 100]))
    return out

def sensitivity_scores(metrics):
    sets = {'paper_5 (20,40,60,80,100)': QUALITY_LEVELS,
            'excl_Q20': [q for q in QUALITY_LEVELS if q != 20],
            'excl_Q100': [q for q in QUALITY_LEVELS if q != 100],
            '3_levels (20,60,100)': [20, 60, 100]}
    if EXTRA_EVAL_LEVELS:
        sets[f'all_{len(ALL_EVAL_LEVELS)}_levels'] = ALL_EVAL_LEVELS
        sets['low_dense (20,30,40,50,60,100)'] = [q for q in ALL_EVAL_LEVELS if q <= 60] + [100]
    res = {}
    for name, lv in sets.items():
        lv = [q for q in lv if q in metrics]
        y = np.array([metrics[q]['mAP50'] for q in lv])
        res[name] = float(y.mean() / y.std(ddof=0)) if y.std(ddof=0) > 0 else float('nan')
    return res

def run_one(method, seed, lib_version):
    key = (method, seed)
    result_json = Path(RESULTS_DIR) / f'{method}_seed{seed}.json'
    if result_json.exists():
        print(f'[completed-skip] {method} seed={seed}')
        return
    print('\n' + '=' * 78 + f'\n▶ {method} | seed={seed} | {datetime.now():%Y-%m-%d %H:%M}\n' + '=' * 78)

    if key in EXISTING_RUNS:
        run_dir = Path(EXISTING_RUNS[key])
        n_train = {'standard': 6471}.get(method, 32355)
        print(f'  reuse an existing runs: {run_dir}')
    else:
        data_yaml, n_train = prepare_data(method, seed)
        print(f'  number of train images: {n_train}')
        run_dir = train_run(method, seed, data_yaml)

    best_pt = Path(run_dir) / 'weights' / 'best.pt'
    assert best_pt.exists(), f'No exist the best.pt: {best_pt}'
    print('  Evaluating ...')
    metrics = evaluate_model(best_pt)
    stats = read_train_stats(run_dir, n_train)
    result = {'method': method, 'seed': seed, 'run_dir': str(run_dir),
              'ultralytics_version': lib_version, 'cls_remap': BASE_TRAIN_ARGS.get('cls_remap'),
              'train_stats': stats,
              'metrics': {str(k): v for k, v in metrics.items()},
              'robustness': robustness_metrics(metrics, QUALITY_LEVELS),
              'sensitivity': sensitivity_scores(metrics)}
    result_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'  Saved: {result_json}  (score_mAP50={result["robustness"]["score_mAP50"]:.3f})')
    if method == 'b3':                                   
        shutil.rmtree(os.path.join(LOCAL_ROOT, f'b3_seed{seed}'), ignore_errors=True)

def run_all():
    lib_version = ensure_env()
    n_runs = len(SEEDS) * len(METHODS)
    est = len(SEEDS) * (1.4 + 6.1 + 6.8)
    print(f'\nTotal {n_runs}runs | about {est:.0f} hr (A100) predicted, '
          f'About ${est * GPU_USD_PER_HOUR:.0f}' + ('  [SMOKE TEST]' if SMOKE_TEST else ''))
    sync_dataset_to_local()
    build_extra_test_levels()
    for seed in SEEDS:                       
        for method in METHODS:
            run_one(method, seed, lib_version)
    print('\nAll processes completed.')

# Summary: mean ± SD, 95% CI, paired t-test, figures
def _load_results():
    rows = []
    for p in sorted(Path(RESULTS_DIR).glob('*_seed*.json')):
        r = json.loads(p.read_text())
        r['metrics'] = {int(k): v for k, v in r['metrics'].items()}
        rows.append(r)
    assert rows, f'no existing the JSON results: {RESULTS_DIR}'
    return rows

def _fmt(m, s, nd=4):
    return f'{m:.{nd}f} ± {s:.{nd}f}' if not (pd.isna(m) or pd.isna(s)) else 'n/a'

def aggregate():
    from scipy import stats as sps
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = _load_results()
    method_order = [m for m in METHODS if any(r['method'] == m for r in runs)]
    label = {'standard': 'Standard (1x, Q100)', 'multiquality': 'Multi-Quality (5x, structured)',
             'b3': 'B3 (5x, random quality)'}

    # (1) Performance by Quality long-table
    long_rows = []
    for r in runs:
        for lv, m in r['metrics'].items():
            long_rows.append({'method': r['method'], 'seed': r['seed'], 'level': lv, **m})
    df = pd.DataFrame(long_rows)
    df.to_csv(f'{SUMMARY_DIR}/per_seed_by_quality.csv', index=False)

    metric_cols = ['mAP50', 'mAP50-95', 'mAP75', 'precision', 'recall']
    g = df.groupby(['method', 'level'])[metric_cols].agg(['mean', 'std', 'count'])
    g.to_csv(f'{SUMMARY_DIR}/summary_by_quality_full.csv')

    tbl = []
    for mth in method_order:
        for lv in ALL_EVAL_LEVELS:
            sub = df[(df.method == mth) & (df.level == lv)]
            if sub.empty:
                continue
            row = {'Model': label[mth], 'Quality (%)': lv, 'n_seeds': len(sub)}
            for c in ('mAP50', 'mAP50-95', 'mAP75'):
                row[c] = _fmt(sub[c].mean(), sub[c].std(ddof=1) if len(sub) > 1 else float('nan'))
            tbl.append(row)
    tbl = pd.DataFrame(tbl)
    tbl.to_csv(f'{SUMMARY_DIR}/summary_mean_sd_by_quality.csv', index=False)

    # (2) robustness index (mean ± SD)
    rb = pd.DataFrame([{'method': r['method'], 'seed': r['seed'], **r['robustness']} for r in runs])
    rb.to_csv(f'{SUMMARY_DIR}/per_seed_robustness.csv', index=False)
    rb_cols = [c for c in rb.columns if c not in ('method', 'seed')]
    rows = []
    for mth in method_order:
        sub = rb[rb.method == mth]
        row = {'Model': label[mth], 'n_seeds': len(sub)}
        for c in rb_cols:
            row[c] = _fmt(sub[c].mean(), sub[c].std(ddof=1) if len(sub) > 1 else float('nan'))
        rows.append(row)
    pd.DataFrame(rows).to_csv(f'{SUMMARY_DIR}/summary_robustness.csv', index=False)

    # (3) Evaluation level composition sensitivity
    sens = pd.DataFrame([{'method': r['method'], 'seed': r['seed'], **r['sensitivity']} for r in runs])
    srows = []
    for mth in method_order:
        sub = sens[sens.method == mth]
        row = {'Model': label[mth]}
        for c in [c for c in sens.columns if c not in ('method', 'seed')]:
            row[c] = _fmt(sub[c].mean(), sub[c].std(ddof=1) if len(sub) > 1 else float('nan'), 2)
        srows.append(row)
    pd.DataFrame(srows).to_csv(f'{SUMMARY_DIR}/sensitivity_robustness_score.csv', index=False)

    # (4) Learning statistics / cost
    st = pd.DataFrame([{'method': r['method'], 'seed': r['seed'], **r['train_stats']} for r in runs])
    st.to_csv(f'{SUMMARY_DIR}/per_seed_training_cost.csv', index=False)
    st_cols = [c for c in st.columns if c not in ('method', 'seed')]
    st.groupby('method')[st_cols].mean().reindex(method_order).to_csv(
        f'{SUMMARY_DIR}/summary_training_cost_mean.csv')

    # (5) Paired comparison: difference between the same seeds -> mean diff, 95% CI, paired t-test
    def value_of(mth, seed, kind, key, lv=None):
        r = next((x for x in runs if x['method'] == mth and x['seed'] == seed), None)
        if r is None:
            return None
        return r['metrics'][lv][key] if kind == 'q' else r['robustness'][key]

    pairs = [('multiquality', 'standard'), ('multiquality', 'b3')]
    prow = []
    items = [('q', k, lv) for k in ('mAP50', 'mAP50-95') for lv in ALL_EVAL_LEVELS] + \
            [('r', k, None) for k in ('score_mAP50', 'auc_mAP50', 'auc_mAP50-95',
                                       'worst_mAP50', 'reldeg_Q20_mAP50', 'sd_mAP50')]
    for a, b in pairs:
        common = sorted({r['seed'] for r in runs if r['method'] == a} &
                        {r['seed'] for r in runs if r['method'] == b})
        if len(common) < 2:
            continue
        for kind, key, lv in items:
            d = np.array([value_of(a, s, kind, key, lv) - value_of(b, s, kind, key, lv) for s in common])
            n, md, sd = len(d), d.mean(), d.std(ddof=1)
            half = sps.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
            p = float(sps.ttest_1samp(d, 0.0).pvalue) if sd > 0 else float('nan')
            prow.append({'comparison': f'{label[a]}  −  {label[b]}', 'metric': key,
                         'level': lv if lv else '-', 'n_seeds': n, 'mean_diff': md, 'sd_diff': sd,
                         'ci95_low': md - half, 'ci95_high': md + half, 'p_paired_t': p,
                         'n_positive': int((d > 0).sum()),
                         'ci_excludes_0': bool((md - half) > 0 or (md + half) < 0)})
    pd.DataFrame(prow).to_csv(f'{SUMMARY_DIR}/paired_comparison.csv', index=False)

    # (6) Figure: mean ± SD by quality
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, key in zip(axes, ('mAP50', 'mAP50-95')):
        for mth in method_order:
            sub = df[df.method == mth].groupby('level')[key].agg(['mean', 'std']).reindex(ALL_EVAL_LEVELS).dropna(subset=['mean'])
            ax.errorbar(sub.index, sub['mean'], yerr=sub['std'].fillna(0), marker='o', capsize=3,
                        linewidth=1.8, label=label[mth])
        ax.set_xlabel('Test quality level (%)')
        ax.set_ylabel(key)
        ax.set_title(f'{key} vs quality (mean ± SD over seeds)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{SUMMARY_DIR}/multiseed_mean_sd.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Summary on console
    pd.set_option('display.width', 200, 'display.max_columns', 30, 'display.max_colwidth', 40)
    print('\n' + '=' * 78 + '\nmAP50-95 (mean ± SD across seeds) by quality\n' + '=' * 78)
    pv = tbl.pivot(index='Model', columns='Quality (%)', values='mAP50-95')
    print(pv.reindex([label[m] for m in method_order]).to_string())
    print('\n' + '=' * 78 + '\nRobustness index (base on mAP50, mean ± SD)\n' + '=' * 78)
    cols = ['score_mAP50', 'mean_mAP50', 'sd_mAP50', 'auc_mAP50', 'worst_mAP50', 'reldeg_Q20_mAP50']
    print(pd.read_csv(f'{SUMMARY_DIR}/summary_robustness.csv').set_index('Model')[['n_seeds'] + cols].to_string())
    pc = pd.read_csv(f'{SUMMARY_DIR}/paired_comparison.csv') if prow else pd.DataFrame()
    if not pc.empty:
        print('\n' + '=' * 78 + '\nCore paired comparison (Multi-Quality − target, 95% CI)\n' + '=' * 78)
        sel = pc[(pc.metric.isin(['score_mAP50'])) | ((pc.metric == 'mAP50-95') & (pc.level.astype(str) == '20'))]
        print(sel[['comparison', 'metric', 'level', 'n_seeds', 'mean_diff', 'ci95_low', 'ci95_high',
                   'p_paired_t', 'ci_excludes_0']].round(4).to_string(index=False))
    print(f'\nSummary results: {SUMMARY_DIR}')


if __name__ == '__main__' or 'get_ipython' in globals():
    run_all()      
    aggregate()
