# main.py — Semantic Segmentation (Single-folder) | OOM-safe + AMP toggle + Accumulation + TQDM + CSV log
# โครงสร้างชุดข้อมูล:
#   <DATASET_ROOT>/
#     train/  -> รูป *.jpg|*.jpeg|*.png (ยกเว้น *_mask.png) + มาสก์คู่ชื่อ *_mask.png
#                _classes.csv (มี/ไม่มี ก็ได้; root/valid/test ก็รองรับ)
#     valid/  -> เช่นเดียวกับ train
#     test/   -> ถ้ามี
#
# .env ตัวอย่าง:
#   RF_API_KEY=...
#   RF_WORKSPACE=hoi-rmqtp
#   RF_PROJECT=human-object-interaction-pcpk1
#   RF_VERSION=2
#   IMGSZ=640
#   BATCH=10
#   EPOCHS=20
#   LR=0.0003
#   ACCUM=8
#   AMP=1                      # 0=ปิด AMP (แนะนำการ์ด 4GB), 1=เปิด AMP
#   FREEZE_BACKBONE_EPOCHS=5   # 0=ไม่ freeze
#   USE_TQDM=1                 # 1=แสดง progress bar
#   LOG_EVERY=10               # ใช้เมื่อตั้ง USE_TQDM=0
#   LOG_FILE=runs/semseg/train_log.csv

import os, sys, csv, random, numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from time import time

# ---- CUDA allocator config (safe on all platforms) ----
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from roboflow import Roboflow
from tqdm import tqdm

# โหลด .env ถ้ามี
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ----------------- Config -----------------
SEED = int(os.getenv("SEED", "42"))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

IMGSZ  = int(os.getenv("IMGSZ", "512"))
BATCH  = int(os.getenv("BATCH", "4"))
EPOCHS = int(os.getenv("EPOCHS", "20"))
LR     = float(os.getenv("LR", "3e-4"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))      # Windows -> 0
ACCUM  = int(os.getenv("ACCUM", "8"))                 # gradient accumulation steps
USE_AMP = bool(int(os.getenv("AMP", "0")))            # 0=fp32, 1=AMP
FREEZE_BACKBONE_EPOCHS = int(os.getenv("FREEZE_BACKBONE_EPOCHS", "5"))
USE_TQDM = bool(int(os.getenv("USE_TQDM", "1")))
LOG_EVERY = int(os.getenv("LOG_EVERY", "10"))
LOG_FILE = os.getenv("LOG_FILE", "runs/semseg/train_log.csv")

def env_required(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v

RF_API_KEY   = env_required("RF_API_KEY")
RF_WORKSPACE = env_required("RF_WORKSPACE")
RF_PROJECT   = env_required("RF_PROJECT")
RF_VERSION   = int(os.getenv("RF_VERSION", "1"))

print(f"Device: {DEVICE}")
print(f"Config -> IMGSZ={IMGSZ} BATCH={BATCH} EPOCHS={EPOCHS} LR={LR} ACCUM={ACCUM} AMP={int(USE_AMP)} FREEZE={FREEZE_BACKBONE_EPOCHS} TQDM={int(USE_TQDM)}")

# ----------------- Download dataset (semantic) -----------------
print("loading Roboflow workspace...")
rf = Roboflow(api_key=RF_API_KEY)
print("loading Roboflow project...")
project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
version = project.version(RF_VERSION)
ds = version.download("png-mask-semantic")
dataset_root = Path(ds.location)
print("Dataset dir ->", dataset_root)

# ----------------- Utils -----------------
def find_split(root: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = root / n
        if p.exists(): return p
    return None

def find_split_dirs(root: Path) -> Tuple[Path, Path, Optional[Path]]:
    train = find_split(root, ["train", "Train"])
    valid = find_split(root, ["valid", "val", "Valid", "Val"])
    test  = find_split(root, ["test", "Test"])
    if train is None or valid is None:
        raise FileNotFoundError(f"Cannot find train/valid under {root}")
    return train, valid, test

def find_classes_csv(root: Path, train: Path, valid: Path, test: Optional[Path]) -> Path:
    candidates = [root/"_classes.csv", train/"_classes.csv", valid/"_classes.csv"]
    if test is not None: candidates.append(test/"_classes.csv")
    for p in candidates:
        if p.exists(): return p
    raise FileNotFoundError("Cannot find _classes.csv (put it in root or train/valid/test).")

def read_classes(csv_path: Path) -> List[str]:
    names: List[str] = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    header = [c.strip().lower() for c in rows[0]] if rows else []
    is_header = any(k in header for k in ["name","class","label"])
    start = 1 if is_header else 0
    for row in rows[start:]:
        for cell in row:
            cell = cell.strip()
            if cell:
                names.append(cell); break
    if not names:
        raise ValueError(f"No class names in {csv_path}")
    return names

# ----------------- Locate splits & classes -----------------
train_dir, valid_dir, test_dir = find_split_dirs(dataset_root)
classes_csv = find_classes_csv(dataset_root, train_dir, valid_dir, test_dir)
class_names = read_classes(classes_csv)
num_classes = 1 + len(class_names)  # 0=background
print(f"Classes ({num_classes}): ['bg', {', '.join(map(str,class_names))}]")

# ----------------- Dataset (single-folder: *_mask.png pairing) -----------------
class SemSegDataset(Dataset):
    """
    โฟลเดอร์เดียว:
      - ภาพ: *.jpg|*.jpeg|*.png (ยกเว้น *_mask.png)
      - มาสก์: <stem>_mask.png  (index mask: 0=bg, 1..C-1)
    ใช้เฉพาะคู่ที่มีครบจริง
    """
    IMG_EXTS = {".jpg", ".jpeg", ".png"}

    def __init__(self, split_dir: Path, img_size=512):
        self.split_dir = Path(split_dir)
        self.img_size = img_size
        self.tf_img = self._build_tf(img_size)

        candidates = [p for p in self.split_dir.iterdir()
                      if p.is_file()
                      and p.suffix.lower() in self.IMG_EXTS
                      and not p.name.endswith("_mask.png")]
        pairs = []
        for img in candidates:
            mask = img.with_name(f"{img.stem}_mask.png")
            if mask.exists():
                pairs.append((img, mask))
        if not pairs:
            raise FileNotFoundError(f"No valid image/mask pairs (*_mask.png) in {self.split_dir}")
        self.pairs = sorted(pairs)

        # preview
        print(f"[{self.split_dir.name}] pairs: {len(self.pairs)}")
        for img, msk in self.pairs[:5]:
            print("  ↳", img.name, "->", msk.name)

    @staticmethod
    def _build_tf(size):
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def set_size(self, new_size: int):
        self.img_size = new_size
        self.tf_img = self._build_tf(new_size)

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        img_path, mask_path = self.pairs[i]
        img = Image.open(img_path).convert("RGB")

        m = Image.open(mask_path)
        if m.mode in ["P", "L"]:
            mask = np.array(m, dtype=np.uint8)
        else:
            # fallback: RGB mask -> map สีเป็น index
            arr = np.array(m)
            h, w = arr.shape[:2]
            flat = arr.reshape(-1, 3)
            colors, inv = np.unique(flat, axis=0, return_inverse=True)
            idx_map = inv.reshape(h, w).astype(np.int32)
            mask = np.clip(idx_map, 0, num_classes-1).astype(np.uint8)

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.fromarray(mask).resize((self.img_size, self.img_size), Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)  # CE loss expects long

        return self.tf_img(img), torch.from_numpy(mask)

# ----------------- Build loaders -----------------
def build_loaders(imgsz, batch):
    train_ds = SemSegDataset(train_dir, imgsz)
    val_ds   = SemSegDataset(valid_dir, imgsz)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_dl   = DataLoader(val_ds,   batch_size=max(1, batch//2), shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    return train_ds, val_ds, train_dl, val_dl

train_ds, val_ds, train_dl, val_dl = build_loaders(IMGSZ, BATCH)
print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

# ----------------- Model (เปลี่ยนตรงนี้ได้บรรทัดเดียว) -----------------
# ค่าเริ่มต้น: LR-ASPP MobileNetV3 (เร็ว/เบา เหมาะ GPU 4GB)
# ตัวเลือกอื่น:
#   models.segmentation.deeplabv3_mobilenet_v3_large(...)
#   models.segmentation.fcn_resnet50(...)
#   models.segmentation.deeplabv3_resnet50(...)
model = models.segmentation.lraspp_mobilenet_v3_large(
    weights=None, num_classes=num_classes
).to(DEVICE)

# (ทางเลือก) freeze backbone ช่วงแรกให้อุ่นเร็วขึ้น
for p in model.backbone.parameters():
    p.requires_grad = True  # จะสลับในลูป

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# AMP API ใหม่ (toggle ได้ด้วย ENV AMP)
scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available() and USE_AMP)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    inter = torch.zeros(num_classes, dtype=torch.float64, device=DEVICE)
    union = torch.zeros(num_classes, dtype=torch.float64, device=DEVICE)
    total_correct = 0; total_pixels = 0

    for imgs, masks in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available() and USE_AMP):
            out = model(imgs)['out']
        preds = out.argmax(1)
        total_correct += (preds == masks).sum().item()
        total_pixels  += masks.numel()
        for c in range(num_classes):
            pred_c = (preds == c); mask_c = (masks == c)
            inter[c] += (pred_c & mask_c).sum()
            union[c] += (pred_c | mask_c).sum()

    miou = (inter / (union + 1e-9)).mean().item()
    pix_acc = total_correct / max(1, total_pixels)
    return miou, pix_acc

# ----------------- Helpers -----------------
def next_down_imgsz(sz: int) -> int:
    for cand in [640, 576, 512, 480, 448, 416, 384, 352, 320, 288, 256]:
        if sz > cand:
            return cand
    return sz

def ensure_log_header(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("epoch,loss,miou,pixacc,imgsz,batch,accum,freeze,amp,time_sec\n")

def append_log(path: Path, epoch:int, loss:float, miou:float, pixacc:float,
               imgsz:int, batch:int, accum:int, freeze:int, amp:int, time_sec:float):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{epoch},{loss:.6f},{miou:.6f},{pixacc:.6f},{imgsz},{batch},{accum},{freeze},{amp},{time_sec:.3f}\n")

# ----------------- OOM-safe Train (ลด batch/imgsz อัตโนมัติ + AMP fallback + TQDM + CSV) -----------------
def try_train(epochs, imgsz, batch, accum):
    global train_ds, val_ds, train_dl, val_dl, USE_AMP

    best_mIoU = -1.0
    patience = int(os.getenv("PATIENCE", "8"))
    bad_epochs = 0
    log_path = Path(LOG_FILE)
    ensure_log_header(log_path)

    for e in range(1, epochs+1):
        model.train()
        # freeze backbone เฉพาะช่วงแรก
        freeze_now = (e <= max(0, FREEZE_BACKBONE_EPOCHS))
        for p in model.backbone.parameters():
            p.requires_grad = not freeze_now

        epoch_loss = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        t0 = time()

        itr = enumerate(train_dl, start=1)
        if USE_TQDM:
            itr = tqdm(itr, total=len(train_dl), ncols=100, leave=False, desc=f"Epoch {e}/{epochs}")

        try:
            for i, (imgs, masks) in itr:
                imgs = imgs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                # AMP autocast + fallback dtype mismatch -> ปิด AMP เฟรมนี้แล้วรัน fp32
                try:
                    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available() and USE_AMP):
                        out = model(imgs)['out']
                        loss = criterion(out, masks) / accum

                    scaler.scale(loss).backward()
                    if i % accum == 0:
                        scaler.step(optimizer); scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                except RuntimeError as amp_e:
                    msg = str(amp_e)
                    if "HalfTensor" in msg and "FloatTensor" in msg:
                        if USE_TQDM: itr.write("[AMP] dtype mismatch -> fallback FP32 (this step)")
                        torch.cuda.empty_cache()
                        out = model(imgs)['out']
                        loss = (criterion(out, masks) / accum)
                        loss.backward()
                        if i % accum == 0:
                            optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    else:
                        raise

                epoch_loss += float(loss.item()) * accum
                steps += 1

                # อัปเดตข้อความใน tqdm หรือพิมพ์ทุก LOG_EVERY
                if USE_TQDM:
                    itr.set_postfix({
                        "loss": f"{(epoch_loss/max(1,steps)):.4f}",
                        "bs": batch,
                        "img": imgsz,
                        "accum": accum,
                        "AMP": int(USE_AMP),
                        "frz": "Y" if freeze_now else "N"
                    })
                elif i % LOG_EVERY == 0:
                    elapsed = time() - t0
                    it_per_s = steps / max(1e-9, elapsed)
                    eta_s = (len(train_dl)-steps) / max(1e-9, it_per_s)
                    print(f"  [e{e}/{epochs} step {i}/{len(train_dl)}] "
                          f"loss={(epoch_loss/max(1,steps)):.4f} | "
                          f"{it_per_s:.2f} it/s | ETA {eta_s:.1f}s | "
                          f"imgsz={imgsz} bs={batch} accum={accum} AMP={int(USE_AMP)} freeze={'Y' if freeze_now else 'N'}",
                          flush=True)

        except RuntimeError as e:
            # กัน OOM: ลด batch → ลด imgsz แล้ว rebuild loaders + retry epoch นี้ใหม่
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg:
                torch.cuda.empty_cache()
                new_batch = max(1, batch // 2) if batch > 2 else batch
                new_imgsz = imgsz if new_batch < batch else next_down_imgsz(imgsz)

                if new_batch == batch and new_imgsz == imgsz:
                    # ลองปิด AMP แล้วรันต่อ ถ้ายังไม่ได้จริง ๆ ค่อยยอมแพ้
                    if USE_AMP:
                        print("[OOM] Disable AMP and retry this epoch.")
                        USE_AMP = False
                        torch.cuda.empty_cache()
                        return try_train(epochs, imgsz, batch, accum)
                    raise

                print(f"[OOM] ลดพารามิเตอร์ -> batch {batch}->{new_batch}, imgsz {imgsz}->{new_imgsz}")
                train_ds, val_ds, train_dl, val_dl = build_loaders(new_imgsz, new_batch)
                return try_train(epochs, new_imgsz, new_batch, accum)
            else:
                raise

        avg_loss = epoch_loss / max(1, steps)
        miou, pix_acc = evaluate(val_dl)
        elapsed_epoch = time() - t0

        print(f"[Epoch {e:03d}/{epochs}] loss={avg_loss:.4f} | val mIoU={miou:.4f} | pixAcc={pix_acc:.4f} | "
              f"imgsz={imgsz} batch={batch} accum={accum} freeze={'Y' if freeze_now else 'N'} AMP={int(USE_AMP)} | "
              f"time={elapsed_epoch:.1f}s",
              flush=True)

        # CSV log
        append_log(Path(LOG_FILE), e, avg_loss, miou, pix_acc, imgsz, batch, accum, int(freeze_now), int(USE_AMP), elapsed_epoch)

        # save best
        save_dir = Path("runs/semseg"); save_dir.mkdir(parents=True, exist_ok=True)
        best_path = save_dir / "best_lraspp_mbv3.pth"
        if miou > best_mIoU:
            best_mIoU = miou
            torch.save({"model": model.state_dict(),
                        "num_classes": num_classes,
                        "class_names": class_names,
                        "imgsz": imgsz}, best_path)
            print(f"  -> Saved best to {best_path} (mIoU={best_mIoU:.4f})")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping (no improvement for {patience} epochs). Best mIoU={best_mIoU:.4f}")
                break

    print("Training finished.")
    return imgsz, batch

# เพิ่มความเสถียรการคูณเมตริกซ์บน GPU consumer
torch.set_float32_matmul_precision("medium")
torch.cuda.empty_cache()

final_imgsz, final_batch = try_train(EPOCHS, IMGSZ, BATCH, ACCUM)

# ----------------- Simple inference -----------------
# ใช้: python main.py <image_or_folder>
if len(sys.argv) > 1:
    src = Path(sys.argv[1])
    if src.is_dir():
        img_paths = sorted([p for p in src.glob("*.*")
                            if p.suffix.lower() in {".jpg",".jpeg",".png"} and not p.name.endswith("_mask.png")])
    else:
        img_paths = [src]

    ckpt_path = Path("runs/semseg/best_lraspp_mbv3.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"]); model.eval()

    size = int(ckpt.get("imgsz", final_imgsz))
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    out_dir = Path("runs/semseg/predict"); out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for p in img_paths:
            im = Image.open(p).convert("RGB")
            t = tf(im).unsqueeze(0).to(DEVICE)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available() and USE_AMP):
                pred = model(t)['out'].argmax(1)[0].detach().cpu().numpy().astype(np.uint8)
            Image.fromarray(pred).save(out_dir / (p.stem + "_mask.png"))
            print("Saved:", out_dir / (p.stem + "_mask.png"))

