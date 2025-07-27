from ultralytics import YOLO

def train_yolo():
    model = YOLO('yolo11n.pt')
    model.train(
        data="C:/Users/akten/Desktop/YOLOv11-Train/revised_data_set-3/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        #resume=True,
        cache=True
    )

if __name__ == "__main__":
    train_yolo()
# This script trains a YOLOv11 model using the specified dataset and parameters.
# Ensure that the YOLOv11 model and dataset path are correctly set before running.