import cv2
import numpy as np
from ultralytics import YOLOWorld

def process_video(source=0):
    # Load a YOLO-World model
    model = YOLOWorld("yolov8s-world.pt")  # or yolov8m-world.pt, yolov8l-world.pt
    
    # Define custom classes
    # We include 'hat', 'cap', 'helmet' to be robust, and 'person' to anchor the detection
    model.set_classes(["person", "hat", "cap"])

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

        # Draw results
        # We iterate over people and check early if they have a cap
        for p_box in person_boxes:
            x1, y1, x2, y2 = map(int, p_box)
            
            # Simple heuristic: Check if any cap box center is near the top of the person box
            # Person head region approx: top 1/5th to be safe? 
            # Or just check if cap box overlaps significantly with the top part.
            
            has_cap = False
            
            # Person width and height
            p_w = x2 - x1
            p_h = y2 - y1
            
            # Define a generous head region (top 25% of the person bbox)
            head_region_y_max = y1 + (p_h * 0.25)
            
            for h_box in headwear_boxes:
                hx1, hy1, hx2, hy2 = map(int, h_box)
                h_center_x = (hx1 + hx2) / 2
                h_center_y = (hy1 + hy2) / 2
                
                # Check if cap center is within the horizontal bounds of person 
                # AND within the vertical head region (allowing it to be slightly above y1 too)
                if (x1 < h_center_x < x2) and (hy2 > y1 - (p_h * 0.1) and hy1 < head_region_y_max):
                    has_cap = True
                    # Optional: Draw the cap box too
                    # cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 1)
                    break
            
            color = (0, 0, 255) # Red for No Cap
            label = "No Cap"
            
            if has_cap:
                color = (0, 255, 0) # Green for Cap
                label = "Wearing Cap"
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Cap Detection", frame)

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
