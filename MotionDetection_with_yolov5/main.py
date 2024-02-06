from update_access_token import update_access_token
from Capture_and_Send import capture_detect, send_immediately, detect, concurrent_execution
import multiprocessing

import cv2 as cv
import queue

#設定を変更したい場合、
# SEND_IMMEDIATELY のbool, 
# Capture_data = capture()
# send_immediately()
# Send_data = send()
#の引数だけ変更する

if __name__ == "__main__":
    Capture_data = capture_detect(save_dir="../../../../1_data_raw", 
                                  w_h=(int(1280), int(720)), 
                                  lsize=(int(320), int(240)), 
                                  mse=int(7), 
                                  WHITE_BALANCE=True, 
                                  awbmode="Auto"
                                  )

    # 撮影完了後すぐに転送したい場合は、True
    # データ転送を行わず、撮影のみを行う場合は、False
    SEND_IMMEDIATELY = True
    # 物体検出を行う場合はTrue, 行わない場合はFalse
    ANALYZE = True

    if SEND_IMMEDIATELY:
        Send_data = send_immediately(dbx_folder_path="Kodera/1_data_raw/TEST")
    elif not SEND_IMMEDIATELY:
        Send_data = None

    if not ANALYZE:
        Analyze_data = None
    elif ANALYZE:
        Analyze_data = detect(
                weights="weights/day_weight/best.pt",  # model path or triton URL
                data= ".yolov5/data/coco128.yaml",  # dataset.yaml path
                imgsz=(640, 640),  # inference size (height, width)
                conf_thres=0.25,  # confidence threshold
                iou_thres=0.45,  # NMS IOU threshold
                max_det=1000,  # maximum detections per image
                device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                view_img=False,  # show results
                save_txt=False,  # save results to *.txt
                save_csv=False,  # save results in CSV format
                save_conf=False,  # save confidences in --save-txt labels
                save_crop=False,  # save cropped prediction boxes
                nosave=False,  # do not save images/videos
                classes=None,  # filter by class: --class 0, or --class 0 2 3
                agnostic_nms=False,  # class-agnostic NMS
                augment=False,  # augmented inference
                visualize=False,  # visualize features
                update=False,  # update all models
                project="../../../..",  # save results to project/name
                name="2_data",  # save results to project/name
                exist_ok=True,  # existing project/name ok, do not increment
                line_thickness=3,  # bounding box thickness (pixels)
                hide_labels=False,  # hide labels
                hide_conf=False,  # hide confidences
                half=False,  # use FP16 half-precision inference
                dnn=False,  # use OpenCV DNN for ONNX inference
                vid_stride=1,  # video frame-rate stride) -> None:
        )


    
    concurrent_execution(Capture_data, 
                         interruption_time="16:38", 
                         restart_time="16:40", 
                         send_immediately=Send_data, 
                         analyze=Analyze_data)
