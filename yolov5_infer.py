import cv2
import time
import ctypes
ctypes.CDLL("./build/libyolo_plugin.so", mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL("./build/libyolo_utils.so", mode=ctypes.RTLD_GLOBAL)
from build import yolov5_trt
    

def draw_detections(image, detections, fps):
    for detection in detections:
        class_id = detection['class_id']
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0x27, 0xC1, 0x36), 2)
        cv2.putText(image, f"{class_id}:{confidence:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0x27, 0xC1, 0x36), 2)
        
    cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
        
    return image

def main(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 
                             fps, (width, height))    

    fps_list = []
    frame_count = 0
    total_time = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        detections = detector.detect(input_image=frame, 
                                     input_w=640, input_h=640, 
                                     conf_thresh=0.45, nms_thresh=0.55)
        
        process_time = time.time() - start_time
        current_fps = 1.0 / process_time if process_time > 0 else 0
        
        frame_count += 1
        total_time += process_time
        fps_list.append(current_fps)

        image = draw_detections(frame, detections, current_fps)
        writer.write(image)

    cap.release()
    writer.release()
    
    if frame_count > 0:
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"Processed {frame_count} frames")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Min FPS: {min(fps_list):.2f}")
        print(f"Max FPS: {max(fps_list):.2f}")


if __name__ == "__main__":
    detector = yolov5_trt.YOLOv5Detector("./weights/yolov5s.engine")
    input_video = "./media/sample_720p.mp4"  
    output_video = "./result.avi"  
    main(input_video, output_video)