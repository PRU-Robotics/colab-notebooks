# roboflow_download.py
# roboflow need to be installed
from roboflow import Roboflow

rf = Roboflow(api_key="xjB7LHpyHPxXRZQuQLiC")
project = rf.workspace("pruida").project("revised_data_set")
version = project.version(3)
dataset = version.download("yolov11")
# The dataset will be downloaded to the current working directory
print(f"Dataset downloaded to: {dataset.location}")
