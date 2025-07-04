import os
import ultralytics
from PIL import Image
from ultralytics import YOLO
from roboflow import Roboflow

# Set working directory
HOME = os.getcwd()
print(f"\nWorking directory: {HOME}\n")

# Check Ultralytics environment
ultralytics.checks()

# NVIDIA GPU Check
os.system("nvidia-smi")

# If needed, create a datasets folder
os.makedirs(f"\n{HOME}/datasets\n", exist_ok=True)

# Replace this with your actual Roboflow API key and project details
rf = Roboflow(api_key="your_roboflow_api_key")
project = rf.workspace("your_workspace_name").project("your_project_name")
version = project.version(2)
dataset = version.download("yolov8")

# Train a YOLOv8 model on the dataset
model = YOLO("yolov8n.yaml")  # or yolov8n.pt if you are fine-tuning a pretrained model

model.train(
    data=os.path.join(HOME, "datasets", "YourDatasetFolder", "data.yaml"),  # TODO: Adjust folder name!
    epochs=3,
    imgsz=640,
    batch=16
)

# Evaluate the model
metrics = model.val()

# Run inference on an image
results = model.predict(
    source="https://media.roboflow.com/notebooks/examples/dog.jpeg", # TODO: path değiştirilecek!
    conf=0.7,
    save=True
)

# Optionally show result image if running in a GUI environment
# Image.open("runs/detect/predict/image.jpg").show()