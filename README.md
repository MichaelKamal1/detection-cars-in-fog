!git clone https://github.com/ultralytics/yolov5  # clone
%cd yolov5
!pip install ultralytics
%pip install -r requirements.txt  # install
%pip install -q roboflow
import torch
import os
from IPython.display import Image, clear_output
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
version = torch.__version__
device_name = torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'

print(f"Setup complete using torch {version} ({device_name}) on device {device}")
from roboflow import Roboflow

# Replace 'your_api_key' with your actual Roboflow API key
api_key = 'IoKLSraDH6kNxC9150yM'

# Initialize Roboflow with the API key
rf = Roboflow(api_key=api_key, model_format='yolov5', notebook='ultralytics')
os.environ['DATASET_DIRECTORY']= '/content/datasets'

rf = Roboflow(api_key="IoKLSraDH6kNxC9150yM")
project = rf.workspace("detection-opjects").project("fog-detection-zfwki")
version = project.version(2)
dataset = version.download("yolov5")
!python train.py --img 416 --epochs 20 --data {dataset.location}/data.yaml --weights /content/yolov5n.pt --cache
%load_ext tensorboard
%tensorboard --logdir runs
from google.colab import files
files.download("/content/yolov5/runs/train/exp2/weights/best.pt")
