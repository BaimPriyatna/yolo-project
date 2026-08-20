import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("ROBOFLOW_API_KEY not found. Set it in .env file.")

rf = Roboflow(api_key=api_key)
project = rf.workspace("ksp-workspace").project("indonesia-license-plate-iqrtj")
version = project.version(2)
dataset = version.download("yolov11")