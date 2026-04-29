import torch
import cv2
import sys

print(f"Versiune Python: {sys.version.split()[0]}")
print(f"Versiune OpenCV: {cv2.__version__}")
print("-" * 30)


if torch.cuda.is_available():
   print(f"Placa video: {torch.cuda.get_device_name(0)}")
else:
   print("Nu se vede placa video")