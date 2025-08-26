import os
import torch
import ultralytics
from ultralytics import YOLO
from roboflow import Roboflow

def main():
    # ================== Setup ==================
    HOME = os.getcwd()
    print(f"\nWorking directory: {HOME}\n")
    print("Ultralytics:", ultralytics.__version__)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # Ortam kontrolü (opsiyonel)
    try:
        ultralytics.checks()
    except Exception as e:
        print("ultralytics.checks() skipped:", e)

    # NVIDIA GPU bilgisi (yoksa sadece geçer)
    os.system("nvidia-smi")

    # datasets klasörü
    DATASETS_DIR = os.path.join(HOME, "datasets")
    os.makedirs(DATASETS_DIR, exist_ok=True)

    # Cihaz seçimi (CUDA varsa 0, yoksa CPU)
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # ================== Roboflow indirme ==================
    # (senin değerlerin)
    rf = Roboflow(api_key="xjB7LHpyHPxXRZQuQLiC")
    project = rf.workspace("pruida").project("detection-djqi0")
    version = project.version(2)

    # Not: Roboflow yolunu kendisi döner (örn: datasets/detection-djqi0-2)
    dataset = version.download("yolov8")  # returns object with .location
    DATA_YAML = os.path.join(dataset.location, "data.yaml")
    print(f"data.yaml: {DATA_YAML}")

    # ================== Model ==================
    # Pretrained ile fine-tune (önerilen)
    model = YOLO("yolov8n.pt")
    # Sıfırdan istersen:
    # model = YOLO("yolov8n.yaml")

    # ================== Train ==================
    # 6 GB VRAM için genelde 8–16 iyi; OOM olursa düşür
    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=16,          # gerekirse 8'e düşür / 32'ye çıkar
        device=DEVICE,
        workers=0,         # <<< Windows freeze sorununu engeller
        deterministic=True,
        project=None,      # runs/detect/train* altına yazar
        name="train"
    )

    # ================== Evaluate ==================
    metrics = model.val(device=DEVICE, workers=0)
    print(metrics)

    # ================== Inference (opsiyonel) ==================
    test_img = os.path.join(HOME, "test.jpg")
    if os.path.exists(test_img):
        results = model.predict(source=test_img, conf=0.5, save=True, device=DEVICE)
        print("Predictions saved under runs/detect/")
    else:
        print("(test.jpg bulunamadı, predict adımı atlandı.)")

if __name__ == "__main__":
    # Windows multiprocessing guard
    import multiprocessing as mp
    mp.freeze_support()
    main()
