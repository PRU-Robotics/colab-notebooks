# YOLOv11 Training - TALAZ USV Perception

This repository contains all the necessary files to train a YOLOv11 object detection model for the TALAZ USV (Unmanned Surface Vehicle) project.

## 🔧 Project Structure

```
yolo_11_train/
├── revised_data_set-3/         # Roboflow-exported dataset
├── yolov11n.pt                 # Pretrained YOLOv11n base model (from Ultralytics releases)
├── roboflow_download.py        # Script to download dataset from Roboflow
├── run_checks.py               # Environment & CUDA availability checker
├── train_yolo11.py             # Custom training script
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-user>/usv-perception.git
cd usv-perception/yolo_11_train
```

> Replace `<your-user>` with your GitHub username if cloned from your fork.

---

### 2. Install dependencies

We recommend using Python **3.10** and a virtual environment.

```bash
pip install ultralytics roboflow
```

To verify CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

### 3. Download dataset from Roboflow

```bash
python roboflow_download.py
```

This will automatically fetch the dataset and extract it into `revised_data_set-3/`.

---

### 4. Start training (on GPU)

```bash
python train_yolo11.py
```

You can modify training parameters such as:

- `epochs=100`
- `imgsz=640`
- `batch=8`
- `device=0` → Use GPU if available

---

## 💻 Colab Alternative

If your local GPU gets too hot, you can train the model on Google Colab.

1. Upload this repository to Colab
2. Run the `roboflow_download.py`
3. Run the `train_yolo11.py`

---

## 🔍 Notes

- Model used: `yolo11n.pt` from official Ultralytics releases.
- Dataset exported from [Roboflow](https://roboflow.com).
- Training performance may vary based on GPU memory and batch size.
- Make sure `runs/` directory exists to store results.

---

## 📦 Output

After training, weights and results will be saved under:

```
runs/detect/train/weights/best.pt
```

You can use this `.pt` file in your inference pipelines.

---

## 👤 Author

**Abdulkerim Akten**  
Piri Reis University - TALAZ USV Team  
[GitHub](https://github.com/abdulkerimakten)
