from detect import run as yolov5_detect  # Assuming you have a callable YOLOv5 function

video_path = 'test.h264'


yolov5_detect(weights="../weights/day_weight/best.pt", 
              source="test.h264")

