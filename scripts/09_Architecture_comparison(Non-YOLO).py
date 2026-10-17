# Non-YOLO Architectures comparison experiment: RT-DETR / Faster R-CNN x Standard / Multi-Quality

import os
import io
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
EXPERIMENT_PATH = os.path.join(GDRIVE_ROOT, 'CCTV_MultiRes_Experiments')
SRC_YOLO_DIR = os.path.join(EXPERIMENT_PATH, 'yolo_dataset')     # existing preprocessed dataset

SMOKE_TEST = False           # True: test run for code check (epochs=1, data 2%)
SEEDS = [0]                  
ARCHS = ['rtdetr', 'fasterrcnn']
METHODS = ['standard', 'multiquality']
QUALITY_LEVELS = [20, 40, 60, 80, 100]
GPU_USD_PER_HOUR = 1.86      
COPY_TO_LOCAL_DISK = True

OUT_DIR = os.path.join('/content/drive/MyDrive/CCTV_MultiRes_Experiments_02_rev', 'arch_generalization' + ('_smoke' if SMOKE_TEST else ''))
RUNS_DIR = os.path.join(OUT_DIR, 'runs')            # Drive: weights/checkpoint
RESULTS_DIR = os.path.join(OUT_DIR, 'results')      # Drive: JSON results by run
SUMMARY_DIR = os.path.join(OUT_DIR, 'summary')      # Drive: summary CSV
LOCAL_ROOT = '/content/work_arch'                   # Colab local SSD
LOCAL_YOLO = os.path.join(LOCAL_ROOT, 'yolo_dataset') if COPY_TO_LOCAL_DISK else SRC_YOLO_DIR

for d in (RUNS_DIR, RESULTS_DIR, SUMMARY_DIR):
    os.makedirs(d, exist_ok=True)

if SMOKE_TEST:
    SEEDS = SEEDS[:1]

CLASS_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van',
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
NUM_CLASSES_YOLO = len(CLASS_NAMES)          
NUM_CLASSES_FRCNN = len(CLASS_NAMES) + 1      

RTDETR_ARGS = dict(
    epochs=1 if SMOKE_TEST else 100, imgsz=640, batch=16,
    optimizer='AdamW', lr0=0.0001, momentum=0.9, weight_decay=0.0001,
    warmup_epochs=3.0, patience=50, workers=8,
    save_period=-1, plots=False, pretrained=True,
    deterministic=True, exist_ok=True, verbose=False,
)
FRCNN_CFG = dict(
    epochs=1 if SMOKE_TEST else 40, 
    batch=8, imgsz=640, lr=0.005, momentum=0.9, weight_decay=0.0005,
    warmup_iters=500, lr_step_size=15, lr_gamma=0.1,
    patience=8, workers=4, conf_thres_pr=0.5, iou_thres_pr=0.5,
)

# Checking the packages and GPU runtime
def ensure_env():
    import importlib
    pkgs = {'ultralytics': 'ultralytics>=8.3.0', 'torchmetrics': 'torchmetrics>=1.4.0'}
    for mod, spec in pkgs.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', spec], check=True)
    import torch
    import ultralytics
    print(f'torch {torch.__version__} | CUDA {torch.cuda.is_available()} | '
          f'ultralytics {ultralytics.__version__}')
    assert torch.cuda.is_available(), 'It is not a GPU runtime (change the runtime type to GPU/A100).'
    return ultralytics.__version__

from tqdm.auto import tqdm  

def _n_workers():
    return max(2, min(16, os.cpu_count() or 4))

def _rsync(src, dst):
    os.makedirs(dst, exist_ok=True)
    if shutil.which('rsync'):
        subprocess.run(['rsync', '-a', src.rstrip('/') + '/', dst.rstrip('/') + '/'], check=True)
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)

def sync_dataset_to_local():
    if not COPY_TO_LOCAL_DISK:
        print('COPY_TO_LOCAL_DISK=False: Skip local copying and use the Drive source directly.')
        return
    subsets = ([('train', q) for q in QUALITY_LEVELS] + [('val', 100)]
               + [('test', q) for q in QUALITY_LEVELS])
    for split, q in tqdm(subsets, desc='dataset copy to local'):
        for kind in ('images', 'labels'):
            src = os.path.join(SRC_YOLO_DIR, kind, split, f'q{q}')
            dst = os.path.join(LOCAL_YOLO, kind, split, f'q{q}')
            if os.path.isdir(dst) and len(os.listdir(dst)) == len(os.listdir(src)):
                continue
            _rsync(src, dst)

def count_images(path):
    n = 0
    for _, _, fs in os.walk(path):
        n += sum(1 for f in fs if f.lower().endswith(('.jpg', '.png')))
    return n

def _list_split_images(split, levels):
    pairs = []
    for q in levels:
        img_dir = os.path.join(LOCAL_YOLO, 'images', split, f'q{q}')
        lbl_dir = os.path.join(LOCAL_YOLO, 'labels', split, f'q{q}')
        for f in sorted(os.listdir(img_dir)):
            if not f.lower().endswith(('.jpg', '.png')):
                continue
            stem = os.path.splitext(f)[0]
            pairs.append((os.path.join(img_dir, f), os.path.join(lbl_dir, stem + '.txt')))
    return pairs

# Image Integrity Check and filtering
def _filter_valid_pairs(pairs, cache_name):
    cache_path = Path(OUT_DIR) / f'bad_images_{cache_name}.json'
    if cache_path.exists():
        bad = set(json.loads(cache_path.read_text()))
    else:
        from PIL import Image

        def check(p):
            try:
                with Image.open(p[0]) as im:
                    im.verify()
                return None
            except Exception:
                return p[0]

        with ThreadPoolExecutor(_n_workers()) as ex:
            results = list(tqdm(ex.map(check, pairs), total=len(pairs),
                                 desc=f'Image Integrity Check ({cache_name})'))
        bad = {r for r in results if r}
        cache_path.write_text(json.dumps(sorted(bad), ensure_ascii=False))
    if bad:
        print(f'  Exclude {len(bad)}ea corrupted/read failed images from the list. ({cache_name}).')
        for p in sorted(bad)[:10]:
            print(f'      - {p}')
        if len(bad) > 10:
            print(f'      ... other {len(bad) - 10}ea (Full list: {cache_path})')
    return [p for p in pairs if p[0] not in bad]

def write_yolo_yaml(name, train_list_or_dir):
    y = {'train': train_list_or_dir,
         'val': os.path.join(LOCAL_YOLO, 'images', 'val', 'q100'),
         'nc': NUM_CLASSES_YOLO, 'names': CLASS_NAMES}
    p = os.path.join(LOCAL_ROOT, f'{name}.yaml')
    with open(p, 'w') as f:
        yaml.dump(y, f, default_flow_style=False)
    return p

def prepare_yolo_style_data(method, tag=''):
    if method == 'standard':
        train = os.path.join(LOCAL_YOLO, 'images', 'train', 'q100')
        return write_yolo_yaml(f'rtdetr_standard{tag}', train), count_images(train)
    if method == 'multiquality':
        paths = []
        for q in QUALITY_LEVELS:                                   
            d = os.path.join(LOCAL_YOLO, 'images', 'train', f'q{q}')
            paths += [os.path.abspath(os.path.join(d, f)) for f in sorted(os.listdir(d))
                      if f.endswith(('.jpg', '.png'))]
        lst = os.path.join(LOCAL_ROOT, f'rtdetr_multiquality{tag}_train.txt')
        with open(lst, 'w') as f:
            f.write('\n'.join(paths))
        return write_yolo_yaml(f'rtdetr_multiquality{tag}', lst), len(paths)
    raise ValueError(method)

# RT-DETR training / evaluation
def rtdetr_train_run(method, seed):
    from ultralytics import RTDETR
    name = f'rtdetr_{method}_seed{seed}'
    run_dir = Path(RUNS_DIR) / name
    done = run_dir / 'train_done.json'
    if done.exists():
        print(f'  [skip] training completed: {name}')
        return run_dir, None
    data_yaml, n_train = prepare_yolo_style_data(method, tag=f'_seed{seed}')
    print(f'  number of train images: {n_train}')
    last = run_dir / 'weights' / 'last.pt'
    t0 = time.time()
    if last.exists():
        print(f'  [continue training] {last}')
        RTDETR(str(last)).train(resume=True)
    else:
        args = dict(RTDETR_ARGS)
        args.update(data=data_yaml, project=RUNS_DIR, name=name, seed=seed)
        RTDETR('rtdetr-l.pt').train(**args)
    done.write_text(json.dumps({'finished_at': datetime.now().isoformat(),
                                'wall_sec_this_session': time.time() - t0}))
    return run_dir, n_train

def rtdetr_evaluate(best_pt):
    from ultralytics import RTDETR
    model = RTDETR(str(best_pt))
    out = {}
    for lv in QUALITY_LEVELS:
        yml = os.path.join(LOCAL_ROOT, f'rtdetr_eval_test_q{lv}.yaml')
        with open(yml, 'w') as f:
            yaml.dump({'train': os.path.join(LOCAL_YOLO, 'images', 'test', f'q{lv}'),
                       'val': os.path.join(LOCAL_YOLO, 'images', 'test', f'q{lv}'),
                       'test': os.path.join(LOCAL_YOLO, 'images', 'test', f'q{lv}'),
                       'nc': NUM_CLASSES_YOLO, 'names': CLASS_NAMES}, f)
        m = model.val(data=yml, split='test', batch=8, imgsz=640, verbose=False, plots=False,
                      project=os.path.join(LOCAL_ROOT, 'val_runs'), name='tmp_rtdetr', exist_ok=True)
        out[lv] = {'mAP50': float(m.box.map50), 'mAP50-95': float(m.box.map),
                   'mAP75': float(m.box.map75),
                   'precision': float(m.box.p.mean()), 'recall': float(m.box.r.mean())}
        print(f'    Q{lv:>3}: mAP50={out[lv]["mAP50"]:.4f}  mAP50-95={out[lv]["mAP50-95"]:.4f}')
    return out

def rtdetr_train_stats(run_dir, n_train):
    df = pd.read_csv(Path(run_dir) / 'results.csv')
    df.columns = [c.strip() for c in df.columns]
    fit_cols = [c for c in df.columns if 'mAP50-95' in c or 'mAP50(B)' in c]
    map5095_col = next(c for c in df.columns if 'mAP50-95' in c)
    map50_col = next(c for c in df.columns if c.endswith('mAP50(B)'))
    fit = 0.1 * df[map50_col] + 0.9 * df[map5095_col]
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
    steps = math.ceil(n_train / RTDETR_ARGS['batch'])
    total_hr = total_sec / 3600.0
    return {'n_train_images': int(n_train), 'epochs_trained': epochs,
            'best_epoch_fitness': int(df['epoch'].iloc[bi]) if 'epoch' in df.columns else bi + 1,
            'steps_per_epoch': steps, 'total_optimizer_updates': steps * epochs,
            'images_seen': int(n_train) * epochs,
            'total_time_hr': total_hr, 'time_to_best_hr': to_best_sec / 3600.0,
            'time_per_epoch_min': total_sec / 60.0 / max(epochs, 1),
            'gpu_cost_usd': total_hr * GPU_USD_PER_HOUR}

# Faster R-CNN training / evaluation (torchvision)
def _build_frcnn_dataset_cls():
    import torch
    from torch.utils.data import Dataset
    from PIL import Image

    class VisDroneTorchDataset(Dataset):

        def __init__(self, pairs, imgsz=640):
            self.pairs = pairs   # [(img_path, label_path), ...]
            self.imgsz = imgsz

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx, _depth=0):
            img_path, lbl_path = self.pairs[idx]
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                if _depth >= 20:
                    raise
                print(f'  image load failure, skip: {img_path} ({e})')
                return self.__getitem__((idx + 1) % len(self.pairs), _depth=_depth + 1)
            w, h = img.size
            boxes, labels = [], []
            if os.path.exists(lbl_path):
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cid, xc, yc, bw, bh = parts
                        cid, xc, yc, bw, bh = int(cid), float(xc), float(yc), float(bw), float(bh)
                        xmin = max(0.0, (xc - bw / 2) * w)
                        ymin = max(0.0, (yc - bh / 2) * h)
                        xmax = min(float(w), (xc + bw / 2) * w)
                        ymax = min(float(h), (yc + bh / 2) * h)
                        if xmax <= xmin or ymax <= ymin:
                            continue
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(cid + 1)   
            import torchvision.transforms.functional as TF
            img_t = TF.to_tensor(img)
            if not boxes:   
                boxes_t = torch.zeros((0, 4), dtype=torch.float32)
                labels_t = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes_t = torch.tensor(boxes, dtype=torch.float32)
                labels_t = torch.tensor(labels, dtype=torch.int64)
            target = {'boxes': boxes_t, 'labels': labels_t, 'image_id': torch.tensor([idx])}
            return img_t, target

    return VisDroneTorchDataset

def _frcnn_collate(batch):
    return tuple(zip(*batch))

def build_frcnn_model():
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights='DEFAULT', min_size=FRCNN_CFG['imgsz'], max_size=FRCNN_CFG['imgsz'])
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES_FRCNN)
    return model

def frcnn_train_run(method, seed):
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    name = f'fasterrcnn_{method}_seed{seed}'
    run_dir = Path(RUNS_DIR) / name
    (run_dir / 'weights').mkdir(parents=True, exist_ok=True)
    done = run_dir / 'train_done.json'
    ckpt_path = run_dir / 'weights' / 'last.pt'
    best_path = run_dir / 'weights' / 'best.pt'
    log_csv = run_dir / 'results.csv'

    levels = [100] if method == 'standard' else QUALITY_LEVELS
    train_pairs = _list_split_images('train', levels)
    val_pairs = _list_split_images('val', [100])
    if SMOKE_TEST:
        train_pairs, val_pairs = train_pairs[:64], val_pairs[:32]
    train_pairs = _filter_valid_pairs(train_pairs, f'train_{method}')
    val_pairs = _filter_valid_pairs(val_pairs, 'val_q100')
    print(f'  number of train images: {len(train_pairs)}')

    if done.exists():
        print(f'  [skip] training completed: {name}')
        return run_dir, len(train_pairs)

    Dataset = _build_frcnn_dataset_cls()
    train_ds, val_ds = Dataset(train_pairs, FRCNN_CFG['imgsz']), Dataset(val_pairs, FRCNN_CFG['imgsz'])
    train_dl = DataLoader(train_ds, batch_size=FRCNN_CFG['batch'], shuffle=True,
                           num_workers=FRCNN_CFG['workers'], collate_fn=_frcnn_collate,
                           drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=FRCNN_CFG['batch'], shuffle=False,
                         num_workers=FRCNN_CFG['workers'], collate_fn=_frcnn_collate)

    device = torch.device('cuda')
    model = build_frcnn_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=FRCNN_CFG['lr'], momentum=FRCNN_CFG['momentum'],
                                 weight_decay=FRCNN_CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=FRCNN_CFG['lr_step_size'], gamma=FRCNN_CFG['lr_gamma'])

    start_epoch, best_map50, bad_epochs, rows = 0, -1.0, 0, []
    if ckpt_path.exists():
        print(f'  [comtinue training] {ckpt_path}')
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epoch'] + 1
        best_map50 = state['best_map50']
        bad_epochs = state.get('bad_epochs', 0)
        if log_csv.exists():
            rows = pd.read_csv(log_csv).to_dict('records')

    n_train = len(train_pairs)
    t0 = time.time()
    for epoch in range(start_epoch, FRCNN_CFG['epochs']):
        model.train()
        ep_t0 = time.time()
        running_loss = 0.0
        n_iter = 0
        for it, (imgs, targets) in enumerate(train_dl):
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if epoch == 0 and it < FRCNN_CFG['warmup_iters']:
                warm_factor = (it + 1) / FRCNN_CFG['warmup_iters']
                for g in optimizer.param_groups:
                    g['lr'] = FRCNN_CFG['lr'] * warm_factor
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_iter += 1
        if epoch == 0:
            for g in optimizer.param_groups:
                g['lr'] = FRCNN_CFG['lr']
        scheduler.step()

        val_metrics = frcnn_evaluate_loader(model, val_dl, device)
        ep_time = time.time() - ep_t0
        row = {'epoch': epoch, 'train_loss': running_loss / max(n_iter, 1),
               'val_mAP50': val_metrics['mAP50'], 'val_mAP50-95': val_metrics['mAP50-95'],
               'time': time.time() - t0, 'lr': optimizer.param_groups[0]['lr']}
        rows.append(row)
        pd.DataFrame(rows).to_csv(log_csv, index=False)
        print(f'  epoch {epoch+1}/{FRCNN_CFG["epochs"]}  loss={row["train_loss"]:.4f}  '
              f'val mAP50={row["val_mAP50"]:.4f}  ({ep_time/60:.1f} min)')

        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'epoch': epoch,
                    'best_map50': best_map50, 'bad_epochs': bad_epochs}, ckpt_path)
        if row['val_mAP50'] > best_map50:
            best_map50, bad_epochs = row['val_mAP50'], 0
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'val_mAP50': best_map50}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= FRCNN_CFG['patience']:
                print(f'  patience({FRCNN_CFG["patience"]}) -> early stopping (epoch {epoch+1})')
                break

    if not best_path.exists():   
        torch.save({'model': model.state_dict(), 'epoch': start_epoch}, best_path)
    done.write_text(json.dumps({'finished_at': datetime.now().isoformat(),
                                'wall_sec_total': time.time() - t0,
                                'epochs_trained': len(rows)}))
    return run_dir, n_train

def frcnn_evaluate_loader(model, data_loader, device):
    import torch
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=False)
    tp = fp = fn = 0
    conf_t, iou_t = FRCNN_CFG['conf_thres_pr'], FRCNN_CFG['iou_thres_pr']
    model.eval()
    with torch.no_grad():
        for imgs, targets in data_loader:
            imgs = [im.to(device) for im in imgs]
            preds = model(imgs)
            preds_cpu = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
            metric.update(preds_cpu, targets)
            for p, t in zip(preds_cpu, targets):
                keep = p['scores'] >= conf_t
                pb, pl = p['boxes'][keep], p['labels'][keep]
                gb, gl = t['boxes'], t['labels']
                matched_gt = set()
                for i in range(len(pb)):
                    best_iou, best_j = 0.0, -1
                    for j in range(len(gb)):
                        if j in matched_gt or int(gl[j]) != int(pl[i]):
                            continue
                        iou = _box_iou(pb[i], gb[j])
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                    if best_iou >= iou_t:
                        tp += 1
                        matched_gt.add(best_j)
                    else:
                        fp += 1
                fn += len(gb) - len(matched_gt)
    r = metric.compute()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {'mAP50': float(r['map_50']), 'mAP50-95': float(r['map']),
            'mAP75': float(r['map_75']), 'precision': float(precision), 'recall': float(recall)}

def _box_iou(a, b):
    xa1, ya1, xa2, ya2 = [float(v) for v in a]
    xb1, yb1, xb2, yb2 = [float(v) for v in b]
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def frcnn_evaluate(best_pt, n_train):
    import torch
    from torch.utils.data import DataLoader
    device = torch.device('cuda')
    model = build_frcnn_model().to(device)
    state = torch.load(best_pt, map_location=device)
    model.load_state_dict(state['model'])
    Dataset = _build_frcnn_dataset_cls()
    out = {}
    for lv in QUALITY_LEVELS:
        pairs = _list_split_images('test', [lv])
        if SMOKE_TEST:
            pairs = pairs[:32]
        pairs = _filter_valid_pairs(pairs, f'test_q{lv}')
        dl = DataLoader(Dataset(pairs, FRCNN_CFG['imgsz']), batch_size=FRCNN_CFG['batch'],
                         shuffle=False, num_workers=FRCNN_CFG['workers'], collate_fn=_frcnn_collate)
        out[lv] = frcnn_evaluate_loader(model, dl, device)
        print(f'    Q{lv:>3}: mAP50={out[lv]["mAP50"]:.4f}  mAP50-95={out[lv]["mAP50-95"]:.4f}')
    return out

def frcnn_train_stats(run_dir, n_train):
    df = pd.read_csv(Path(run_dir) / 'results.csv')
    bi = int(df['val_mAP50'].values.argmax())
    epochs = int(len(df))
    total_sec = float(df['time'].iloc[-1])
    to_best_sec = float(df['time'].iloc[bi])
    steps = math.ceil(n_train / FRCNN_CFG['batch'])
    total_hr = total_sec / 3600.0
    return {'n_train_images': int(n_train), 'epochs_trained': epochs,
            'best_epoch_fitness': int(df['epoch'].iloc[bi]),
            'steps_per_epoch': steps, 'total_optimizer_updates': steps * epochs,
            'images_seen': int(n_train) * epochs,
            'total_time_hr': total_hr, 'time_to_best_hr': to_best_sec / 3600.0,
            'time_per_epoch_min': total_sec / 60.0 / max(epochs, 1),
            'gpu_cost_usd': total_hr * GPU_USD_PER_HOUR}

# Robustness index
def robustness_metrics(metrics, levels):
    lv = sorted(levels)
    out = {}
    for key, tag in (('mAP50', 'mAP50'), ('mAP50-95', 'mAP50-95')):
        y = np.array([metrics[q][key] for q in lv])
        out[f'mean_{tag}'] = float(y.mean())
        out[f'sd_{tag}'] = float(y.std(ddof=0))
        out[f'score_{tag}'] = float(y.mean() / y.std(ddof=0)) if y.std(ddof=0) > 0 else float('nan')
    return out

def run_one(arch, method, seed, lib_version):
    key = (arch, method, seed)
    result_json = Path(RESULTS_DIR) / f'{arch}_{method}_seed{seed}.json'
    if result_json.exists():
        print(f'[completed-skip] {arch} / {method} seed={seed}')
        return
    print('\n' + '=' * 78 +
          f'\n {arch} | {method} | seed={seed} | {datetime.now():%Y-%m-%d %H:%M}\n' + '=' * 78)

    if arch == 'rtdetr':
        run_dir, n_train = rtdetr_train_run(method, seed)
        best_pt = Path(run_dir) / 'weights' / 'best.pt'
        assert best_pt.exists(), f'not existing the best.pt: {best_pt}'
        print('  evaluating ...')
        metrics = rtdetr_evaluate(best_pt)
        stats = rtdetr_train_stats(run_dir, n_train)
    elif arch == 'fasterrcnn':
        run_dir, n_train = frcnn_train_run(method, seed)
        best_pt = Path(run_dir) / 'weights' / 'best.pt'
        assert best_pt.exists(), f'not existing the best.pt: {best_pt}'
        print('  evaluating ...')
        metrics = frcnn_evaluate(best_pt, n_train)
        stats = frcnn_train_stats(run_dir, n_train)
    else:
        raise ValueError(arch)

    result = {'arch': arch, 'method': method, 'seed': seed, 'run_dir': str(run_dir),
              'lib_version': lib_version, 'train_stats': stats,
              'metrics': {str(k): v for k, v in metrics.items()},
              'robustness': robustness_metrics(metrics, QUALITY_LEVELS)}
    result_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'  Saved: {result_json}  (score_mAP50={result["robustness"]["score_mAP50"]:.3f})')

def run_all():
    lib_version = ensure_env()
    n_runs = len(SEEDS) * len(ARCHS) * len(METHODS)
    print(f'\nTotal {n_runs}runs training' + ('  [SMOKE TEST]' if SMOKE_TEST else ''))
    sync_dataset_to_local()
    for seed in SEEDS:
        for arch in ARCHS:
            for method in METHODS:
                run_one(arch, method, seed, lib_version)
    print('\nAll processes running completed.')

# Summary
ARCH_LABEL = {'rtdetr': 'RT-DETR', 'fasterrcnn': 'FasterRCNN'}
METHOD_LABEL = {'standard': 'Standard', 'multiquality': 'MultiQuality'}

def _load_results():
    rows = []
    for p in sorted(Path(RESULTS_DIR).glob('*_seed*.json')):
        r = json.loads(p.read_text())
        r['metrics'] = {int(k): v for k, v in r['metrics'].items()}
        rows.append(r)
    assert rows, f'There is no result JSON: {RESULTS_DIR}'
    return rows

def aggregate():
    rows = _load_results()
    by_key = {}
    for r in rows:
        by_key.setdefault((r['arch'], r['method']), []).append(r)

    comp_rows, rob_rows, stat_rows = [], [], []
    for (arch, method), runs in sorted(by_key.items()):
        model_name = f'{ARCH_LABEL[arch]}-{METHOD_LABEL[method]}'
        # mAP50/mAP50-95/mAP75/precision/recall by quality (mean)
        for q in QUALITY_LEVELS:
            keys = ['mAP50', 'mAP50-95', 'mAP75', 'precision', 'recall']
            vals = {k: np.mean([r['metrics'][q][k] for r in runs]) for k in keys}
            comp_rows.append({'Model': model_name, 'Quality (%)': q, **vals})
        # Robustness Score (mAP50 mean/SD across 5 levels, mean)
        mean50 = np.mean([r['robustness']['mean_mAP50'] for r in runs])
        sd50 = np.mean([r['robustness']['sd_mAP50'] for r in runs])
        score50 = mean50 / sd50 if sd50 > 0 else float('nan')
        rob_rows.append({'Model': model_name, 'mean_mAP50': mean50, 'sd_mAP50': sd50,
                          'robustness_score': score50})
        # summary
        for r in runs:
            s = r['train_stats']
            stat_rows.append({'Model': model_name, 'seed': r['seed'],
                               'Total Epoch': s['epochs_trained'],
                               'Total Time (hr)': round(s['total_time_hr'], 3),
                               'Best Epoch': s['best_epoch_fitness'],
                               'Time to Reach Peak (hr)': round(s['time_to_best_hr'], 3),
                               'GPU Cost (USD)': round(s['gpu_cost_usd'], 2),
                               'N Train Images': s['n_train_images']})

    comp_df = pd.DataFrame(comp_rows)
    rob_df = pd.DataFrame(rob_rows)
    stat_df = pd.DataFrame(stat_rows)

    comp_path = os.path.join(SUMMARY_DIR, 'architecture_comparison_table_F.csv')
    rob_path = os.path.join(SUMMARY_DIR, 'architecture_robustness_scores_F.csv')
    stat_path = os.path.join(SUMMARY_DIR, 'architecture_training_stats_F.csv')
    comp_df.to_csv(comp_path, index=False)
    rob_df.to_csv(rob_path, index=False)
    stat_df.to_csv(stat_path, index=False)

    print(f'saved: {comp_path}')
    print(f'saved: {rob_path}')
    print(f'saved: {stat_path}')
    print('\n=== Robustness Score (mAP50) ===')
    print(rob_df.to_string(index=False))
    return comp_df, rob_df, stat_df


if __name__ == '__main__':
    run_all()
    aggregate()


# Visualization
import os
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Directory path setup
RESULT_DIR = '/content/drive/MyDrive/CCTV_MultiRes_Experiments/arch_generalization/summary'              # directory for architecture_comparison_table_F.csv , etc.
OLD_RESULT_DIR = '/content/drive/MyDrive/CCTV_MultiRes_Experiments/evaluation_results'                   # directory for existing YOLO result CSV file 
OUT_DIR = os.path.join(RESULT_DIR, 'figures2')
MERGE_EXISTING_YOLO = True    # Whether to merge with the existing YOLOv8m/YOLOv8n/YOLO11m results

os.makedirs(OUT_DIR, exist_ok=True)

ARCH_COLOR = {
    'YOLOv8m':    '#2a78d6',   # blue
    'YOLOv8n':    '#eb6834',   # orange
    'YOLO11m':    '#1baf7a',   # aqua
    'RT-DETR':    '#eda100',   # yellow
    'FasterRCNN': '#e87ba4',   # magenta
}
ARCH_ORDER = ['YOLOv8m', 'YOLOv8n', 'YOLO11m', 'RT-DETR', 'FasterRCNN']
METHOD_ORDER = ['Standard', 'MultiQuality']
METHOD_STYLE = {
    'Standard':     dict(linestyle='-',  marker='o'),
    'MultiQuality': dict(linestyle='--', marker='s'),
}
METHOD_LABEL_KR = {'Standard': 'Standard', 'MultiQuality': 'Multi-Quality'}
METHOD_BAR_COLOR = {'Standard': '#2a78d6', 'MultiQuality': '#eb6834'}

mpl.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
    'axes.edgecolor': '#52514e', 'text.color': '#0b0b0b',
    'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

def normalize_model(name):
    name = str(name).strip()
    if name == 'Standard (Q100)':
        return 'YOLOv8m', 'Standard'
    if name == 'Multi-Quality':
        return 'YOLOv8m', 'MultiQuality'
    for method in ('MultiQuality', 'Standard'):
        suffix = '-' + method
        if name.endswith(suffix):
            return name[: -len(suffix)], method
    raise ValueError(f"The model name cannot be interpreted as (architecture, method): '{name}'")

def _find_csv(result_dir, filename):
    p = os.path.join(result_dir, filename)
    if os.path.exists(p):
        return p
    hits = glob.glob(os.path.join(result_dir, '**', filename), recursive=True)
    return hits[0] if hits else None

def load_comparison_table():
    f_path = _find_csv(RESULT_DIR, 'architecture_comparison_table_F.csv')
    assert f_path, (f"Could not find 'architecture_comparison_table_F.csv' under {RESULT_DIR}. Check RESULT_DIR.")
    df = pd.read_csv(f_path)
    print(f'load: {f_path} ({len(df)} row)')

    if MERGE_EXISTING_YOLO:
        old_path = _find_csv(OLD_RESULT_DIR, 'architecture_comparison_table.csv')
        if old_path:
            old = pd.read_csv(old_path)
            df = pd.concat([old, df], ignore_index=True)
            print(f'merge: {old_path} ({len(old)} row) -> total {len(df)} row')
        else:
            print('Since the existing YOLO results (architecture_comparison_table.csv) could not be found, only the F results are used.')

    arch, method = zip(*df['Model'].map(normalize_model))
    df = df.assign(Arch=arch, Method=method)
    return df

def load_robustness_scores():
    f_path = _find_csv(RESULT_DIR, 'architecture_robustness_scores_F.csv')
    assert f_path, (f"Could not find 'architecture_robustness_scores_F.csv' under {RESULT_DIR}. Check RESULT_DIR.")
    df = pd.read_csv(f_path)
    print(f'load: {f_path} ({len(df)} row)')

    if MERGE_EXISTING_YOLO:
        old_path = _find_csv(OLD_RESULT_DIR, 'architecture_robustness_scores.csv')
        if old_path:
            old = pd.read_csv(old_path)
            df = pd.concat([old, df], ignore_index=True)
            print(f'merge: {old_path} ({len(old)} row) -> total {len(df)} row')

    arch, method = zip(*df['Model'].map(normalize_model))
    df = df.assign(Arch=arch, Method=method)
    return df


def _present_archs(df):
    have = set(df['Arch'].unique())
    return [a for a in ARCH_ORDER if a in have]


def plot_metric_vs_quality(df, metric, ylabel, out_name):
    archs = _present_archs(df)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for arch in archs:
        for method in METHOD_ORDER:
            sub = df[(df['Arch'] == arch) & (df['Method'] == method)].sort_values('Quality (%)')
            if sub.empty:
                continue
            style = METHOD_STYLE[method]
            ax.plot(sub['Quality (%)'], sub[metric], color=ARCH_COLOR[arch],
                     linewidth=2.0, markersize=6, markerfacecolor=(
                         ARCH_COLOR[arch] if method == 'Standard' else 'white'),
                     markeredgecolor=ARCH_COLOR[arch], markeredgewidth=1.4, **style)

    ax.set_xlabel('Test Quality Level (%)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(df['Quality (%)'].unique()))
    ax.set_ylim(bottom=0)

    arch_handles = [plt.Line2D([0], [0], color=ARCH_COLOR[a], linewidth=2.5) for a in archs]
    legend1 = ax.legend(arch_handles, archs, title='Architecture', loc='lower right',
                         frameon=False, fontsize=9, title_fontsize=9)
    ax.add_artist(legend1)
    method_handles = [plt.Line2D([0], [0], color='#52514e', markerfacecolor=(
                        '#52514e' if m == 'Standard' else 'white'),
                        markeredgecolor='#52514e', **METHOD_STYLE[m])
                       for m in METHOD_ORDER]
    ax.legend(method_handles, [METHOD_LABEL_KR[m] for m in METHOD_ORDER], title='Method',
              loc='lower left', frameon=False, fontsize=9, title_fontsize=9)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_DIR, f'{out_name}.{ext}'))
    plt.close(fig)
    print(f'저장: {out_name}.png / .pdf')


def plot_robustness_bar(rob_df, out_name='fig_robustness_score'):
    archs = _present_archs(rob_df)
    x = np.arange(len(archs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for i, method in enumerate(METHOD_ORDER):
        vals = []
        for arch in archs:
            row = rob_df[(rob_df['Arch'] == arch) & (rob_df['Method'] == method)]
            vals.append(float(row['robustness_score'].iloc[0]) if not row.empty else np.nan)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width=width, color=METHOD_BAR_COLOR[method],
                       label=METHOD_LABEL_KR[method])
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, 0.15, f'{v:.2f}',
                        ha='center', va='bottom', fontsize=8.5, color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(archs)
    ax.set_ylabel('Robustness Score (mean mAP50 / SD, Eq. 5)')
    ax.legend(title='Method', frameon=False, fontsize=9, title_fontsize=9)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_DIR, f'{out_name}.{ext}'))
    plt.close(fig)
    print(f'saved: {out_name}.png / .pdf')

def plot_robustness_improvement(rob_df, out_name='fig_robustness_improvement'):
    archs = _present_archs(rob_df)
    improvements = []
    for arch in archs:
        std = rob_df[(rob_df['Arch'] == arch) & (rob_df['Method'] == 'Standard')]
        mq = rob_df[(rob_df['Arch'] == arch) & (rob_df['Method'] == 'MultiQuality')]
        if std.empty or mq.empty:
            improvements.append(np.nan)
            continue
        s, m = float(std['robustness_score'].iloc[0]), float(mq['robustness_score'].iloc[0])
        improvements.append((m / s - 1.0) * 100.0 if s > 0 else np.nan)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bars = ax.bar(archs, improvements, color=[ARCH_COLOR[a] for a in archs], width=0.55)
    for b, v in zip(bars, improvements):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + (1.5 if v >= 0 else -3.5),
                     f'{v:+.1f}%', ha='center', va='bottom' if v >= 0 else 'top',
                     fontsize=9.5, color='#0b0b0b')
    ax.axhline(0, color='#52514e', linewidth=0.8)
    ax.set_ylabel('Robustness Score Improvement\n(Multi-Quality vs. Standard, %)')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_DIR, f'{out_name}.{ext}'))
    plt.close(fig)
    print(f'saved: {out_name}.png / .pdf')

def save_summary_table(comp_df, rob_df):
    rows = []
    for arch in _present_archs(comp_df):
        for method in METHOD_ORDER:
            c = comp_df[(comp_df['Arch'] == arch) & (comp_df['Method'] == method)]
            r = rob_df[(rob_df['Arch'] == arch) & (rob_df['Method'] == method)]
            if c.empty or r.empty:
                continue
            q100 = c[c['Quality (%)'] == 100]
            rows.append({
                'Architecture': arch, 'Method': METHOD_LABEL_KR[method],
                'mAP50 @ Q100': round(float(q100['mAP50'].iloc[0]), 4) if not q100.empty else np.nan,
                'mean mAP50 (Q20-100)': round(float(r['mean_mAP50'].iloc[0]), 4),
                'SD mAP50': round(float(r['sd_mAP50'].iloc[0]), 4),
                'Robustness Score': round(float(r['robustness_score'].iloc[0]), 3),
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, 'summary_table.csv'), index=False)
    print('\n=== Summary Table ===')
    print(out.to_string(index=False))
    return out

def main():
    comp_df = load_comparison_table()
    rob_df = load_robustness_scores()

    plot_metric_vs_quality(comp_df, 'mAP50', 'mAP50', 'fig_map50_vs_quality')
    plot_metric_vs_quality(comp_df, 'mAP50-95', 'mAP50-95', 'fig_map50_95_vs_quality')
    plot_robustness_bar(rob_df)
    plot_robustness_improvement(rob_df)
    save_summary_table(comp_df, rob_df)
    print(f'\nAll images were saved to {OUT_DIR}.')


if __name__ == '__main__':
    main()