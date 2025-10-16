# 1) ติดตั้งไลบรารีหลัก
!pip install -q torch torchvision roboflow python-dotenv tqdm

# 2) โคลน repo ของคุณ
!git clone https://github.com/wongsakron/Semantic-Segmentation.git
%cd Semantic-Segmentation

# 3) สร้างไฟล์ .env (ใส่ค่า Roboflow และ training config)
env_text= """RF_API_KEY=80Aajg90uyCx2KOrY2i8
RF_WORKSPACE=hoi-rmqtp
RF_PROJECT=human-object-interaction-pcpk1
RF_VERSION=2
IMGSZ=640
BATCH=12
EPOCHS=30
LR=0.0004
ACCUM=3
AMP=1
PRELOAD=1
FREEZE_BACKBONE_EPOCHS=1
NUM_WORKERS=2
COMPILE=0
USE_TQDM=1
LOG_EVERY=10
LOG_FILE=runs/semseg/train_log.csv
"""

open('.env','w').write(env_text)

# 4) รันแบบไม่บัฟเฟอร์เพื่อดู log ต่อเนื่อง
!python -u main.py
