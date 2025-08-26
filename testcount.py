import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Load COCO class list
with open("C:/Users/vigneswar/Downloads/SPS Sample/yolov8parkingspace/coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Define parking areas (12 slots)
areas = [
    [(52, 364), (30, 417), (73, 412), (88, 369)],
    [(105, 353), (86, 428), (137, 427), (146, 358)],
    [(159, 354), (150, 427), (204, 425), (203, 353)],
    [(217, 352), (219, 422), (273, 418), (261, 347)],
    [(274, 345), (286, 417), (338, 415), (321, 345)],
    [(336, 343), (357, 410), (409, 408), (382, 340)],
    [(396, 338), (426, 404), (479, 399), (439, 334)],
    [(458, 333), (494, 397), (543, 390), (495, 330)],
    [(511, 327), (557, 388), (603, 383), (549, 324)],
    [(564, 323), (615, 381), (654, 372), (596, 315)],
    [(616, 316), (666, 369), (703, 363), (642, 312)],
    [(674, 311), (730, 360), (764, 355), (707, 308)]
]

# Initialize OpenCV window
cv2.namedWindow('ParkVision')

# Mouse callback for debugging coordinates
def ParkVision(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse coordinates: {x}, {y}")

cv2.setMouseCallback('ParkVision', ParkVision)

# Load video
cap = cv2.VideoCapture('parking1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    # Resize the frame for consistent display
    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection
    results = model.predict(frame, verbose=False)
    detections = results[0].boxes.data if len(results[0].boxes) > 0 else []
    df = pd.DataFrame(detections, columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class_id']).astype("float")

    occupied = [0] * len(areas)  # Track occupancy for each area
    cars_found = False  # Flag to check if any car is found

    for _, row in df.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row[:6])
        label = class_list[class_id] if class_id < len(class_list) else "Unknown"

        # Check if the detected object is a car
        if label == 'car':
            cars_found = True
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Calculate center of the bounding box

            for i, area in enumerate(areas):
                if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                    occupied[i] = 1
                    # Annotate the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    # Calculate free and filled slots
    filled = sum(occupied)
    free = len(areas) - filled

    # Draw parking area outlines and labels
    for i, area in enumerate(areas):
        color = (0, 0, 255) if occupied[i] else (0, 255, 0)
        cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
        pos = area[1]  # Position of the slot label
        cv2.putText(frame, str(i + 1), pos, cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

    # Display filled and free spots count (updated order)
    cv2.putText(frame, f"Spots Filled: {filled}", (23, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.putText(frame, f"Spots Left: {free}", (23, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("ParkVision", frame)

    # Adjust playback speed dynamically
    delay = 200 if cars_found else 30  # Slower when cars are found
    if cv2.waitKey(delay) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
