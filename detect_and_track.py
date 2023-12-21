import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
import os

MODEL = "runs/detect/train5/weights/best.pt"
# MODEL = "yolov8s.pt"
# class_ids of interest
CLASS_ID = [1, 2, 3, 4, 5, 6, 7]
# Specify the path to your video file
video_path = "test_child.mp4"

def detections_list(detections:sv.Detections, CLASS_NAMES_DICT):
    # create a list variable that will contain the info for all detected objects of one frame
    """ Create a list variable that contains info for all detected objects within a frame.
    
        Input:
        
            detections: detections of a single frame
            CLASS_NAMES_DICT: mapping between class number and class name
            
        Return:

            This function returns a list of dictionaries. [{},{}..]
            detect_list[0] contains the information of the first detected object in the following format:
            {
                Track ID: 1
                Class: Child
                Confidence: 0.80
            }

    """
    detect_list = []
    
    for i in range(len(detections)):
        # create a dictionary variable that will contain the info of one detected object
        detect_dict = {}
        
        confidence = detections.confidence[i] if detections.confidence is not None else None
        class_id = detections.class_id[i] if detections.class_id is not None else None
        tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else None

        if tracker_id is not None:       
            detect_dict['Track ID: '] = tracker_id
        if class_id is not None:       
            detect_dict['Class: '] = CLASS_NAMES_DICT[class_id]
        if confidence is not None:
            detect_dict['Confidence: '] = confidence
           
        detect_list.append(detect_dict)
    
    return detect_list
    

def process_video_frame(video_path, model=YOLO(MODEL)):
    """
    This function takes a path to a video file and returns each frame annotated 
    along with a list object containing information about the detected objects

        Input:
            video_path: path to a video file
            model: YOLO model
            
        Returns:
            Returns a single annotated frame along with a list object
            containing information about the detected objects
    
    """
    
    model.fuse()
    CLASS_NAMES_DICT = model.model.names

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # create VideoInfo, ByteTracker and BoxAnnotator instance
    video_info = sv.VideoInfo.from_video_path(video_path)    
    byte_tracker = sv.ByteTrack(track_thresh=0.15, match_thresh= 0.2, frame_rate=video_info.fps)
    box_annotator = sv.BoxAnnotator()
    
    # initial setting of the variables
    count = 0
    detections = sv.Detections.empty()
    labels = []

    try:
        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # Check if it was successfully read
            if not ret:
                break
            
            count += 1
        
            if count % 6 == 1: 
                # model prediction on single frame and conversion to supervision Detections
                results = model(frame)[0]
                
            detections = sv.Detections.from_ultralytics(results)
            
            # filter detections            
            filtered_detections = sv.Detections.empty()
            for i in range(len(detections)):
                if detections[i].class_id in CLASS_ID:
                    filtered_detections = sv.Detections.merge([filtered_detections, detections[i]])
                else: continue
                
            # tracking detections
            filtered_detections = byte_tracker.update_with_detections(detections=filtered_detections)
            
            labels = [
                f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _ in filtered_detections
            ]
            # annotate and write frame
            frame = box_annotator.annotate(scene=frame, detections=filtered_detections, labels=labels)
            detect_list = detections_list(filtered_detections, CLASS_NAMES_DICT)
            
  
            # Yield the frame for further processing if needed
            yield frame, detect_list

    finally:
        # Release the video capture object and close any OpenCV windows
        cap.release()







# Iterate over the frames and process them
for frame, detect_list in process_video_frame(video_path):
   
    cv2.imshow('main',frame)
    for detection in detect_list:
        print(detection)
    if cv2.waitKey(33) == ord('q'):
        break
    
cv2.destroyAllWindows()
# When the loop exits, the video has been fully processed
