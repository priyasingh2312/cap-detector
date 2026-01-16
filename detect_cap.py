import cv2
import numpy as np
from ultralytics import YOLOWorld

def process_video(source=0):
    # Load a YOLO-World model
    model = YOLOWorld("yolov8s-world.pt")  # or yolov8m-world.pt, yolov8l-world.pt
    
    # Define custom classes
    # We include 'hat', 'cap', 'helmet' to be robust, and 'person' to anchor the detection
    # 'button' is added for shirt button detection
    model.set_classes(["person", "hat", "cap", "button"])

    # Open video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model.predict(frame, conf=0.1, verbose=False)
        result = results[0] # First image

        # Parse detections
        person_boxes = []
        headwear_boxes = []
        button_boxes = []

        # results.boxes.cls contains class indices
        # results.names maps indices to class names
        
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            xyxy = box.xyxy[0]
            
            if cls_name == "person":
                person_boxes.append(xyxy)
            elif cls_name in ["hat", "cap"]:
                headwear_boxes.append(xyxy)
            elif cls_name == "button":
                button_boxes.append(xyxy)

        # Draw results
        # We iterate over people and check for attributes
        for p_box in person_boxes:
            x1, y1, x2, y2 = map(int, p_box)
            
            # --- CAP DETECTION PROPERTIES ---
            # Person width and height
            p_w = x2 - x1
            p_h = y2 - y1
            
            # Define a generous head region (top 25% of the person bbox)
            head_region_y_max = y1 + (p_h * 0.25)
            
            has_cap = False
            for h_box in headwear_boxes:
                hx1, hy1, hx2, hy2 = map(int, h_box)
                h_center_x = (hx1 + hx2) / 2
                h_center_y = (hy1 + hy2) / 2
                
                # Check if cap center is within person bounds and head region
                if (x1 < h_center_x < x2) and (hy2 > y1 - (p_h * 0.1) and hy1 < head_region_y_max):
                    has_cap = True
                    break
            
            # --- BUTTON DETECTION PROPERTIES ---
            # Define torso region (approx 20% to 70% of height)
            torso_y_min = y1 + (p_h * 0.2)
            torso_y_max = y1 + (p_h * 0.7)
            
            has_buttons = False
            for b_box in button_boxes:
                bx1, by1, bx2, by2 = map(int, b_box)
                b_center_x = (bx1 + bx2) / 2
                b_center_y = (by1 + by2) / 2
                
                # Check if button is within person horizontal bounds and torso vertical bounds
                if (x1 < b_center_x < x2) and (torso_y_min < b_center_y < torso_y_max):
                    has_buttons = True
                    # Optional: Draw small circle for button
                    cv2.circle(frame, (int(b_center_x), int(b_center_y)), 3, (255, 0, 0), -1)

            # --- DISPLAY ---
            cap_color = (0, 255, 0) if has_cap else (0, 0, 255)
            cap_text = "Wearing Cap" if has_cap else "No Cap"
            
            button_color = (255, 255, 0) if has_buttons else (0, 165, 255) # Cyan if yes, Orange if no
            button_text = "Buttons: Yes" if has_buttons else "Buttons: No"

            cv2.rectangle(frame, (x1, y1), (x2, y2), cap_color, 2)
            
            # Draw texts
            cv2.putText(frame, cap_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cap_color, 2)
            cv2.putText(frame, button_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, button_color, 2)

        cv2.imshow("Cap & Button Detection", frame)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect if people are wearing caps in video.")
    parser.add_argument("--source", type=str, default="0", help="Video source: '0' for webcam or path to video file.")
    args = parser.parse_args()
    
    # helper to convert numeric string to int for webcam
    src = args.source
    if src.isdigit():
        src = int(src)
        
    process_video(src)
