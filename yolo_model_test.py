import cv2
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from pathlib import Path

def main():

    base_dir = Path(__file__).resolve().parent
    model_path = base_dir/"best.pt"
    model = YOLO(str(model_path))

    test_imgs_path = base_dir/"test_img"
    if not os.path.exists(test_imgs_path):
        print("error")

    print("successed")






if __name__ == "__main__":
    main()