import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

# Define the function before using it
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

# Load YOLO model
model = YOLO('yolov8s.pt')

# Create window
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('parking1.mp4')

# Open coco.txt file
my_file = open('C:/Users/vigneswar/Downloads/SPS Sample/yolov8parkingspace/coco.txt', 'r')
data = my_file.read()
class_list = data.split("\n")

# Example area definition
area9 = [(511, 327), (557, 388), (603, 383), (549, 324)]

# Placeholder for video processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        input()
        print("End of video")
        break
    
    cv2.imshow('RGB', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
