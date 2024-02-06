###########################################################
# 
# PCS_v9 is a program for rasberry pi camera only.
# Keep in mind that this program can't use USB camera and opencv.
#
###########################################################

"""
####  picamera2 License  ####

BSD 2-Clause License

Copyright (c) 2021, Raspberry Pi
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput
from libcamera import controls


import sched, time
import numpy as np
import cv2 as cv
import os
import datetime
#import concurrent.futures as Executor
import concurrent.futures
import sys
import dropbox
import glob
import requests
from pathlib import Path
#import multiprocessing
import queue


from update_access_token import update_access_token
#from get_file_creation_time import get_file_creation_time
from keyboard_wait import keyboard_wait_repeat
#from lock_directory import lock_directory, unlock_directory
from decorator import error_handler



# This motion detected program is based on picamera2/examples/capture_circular.py
# <https://github.com/raspberrypi/picamera2/blob/main/examples/capture_circular.py>

# you can use HDR mode by typing the following into terminal window before starting Picamera2
## v4l2-ctl --set-ctrl wide_dynamic_range=1 -d /dev/v4l-subdev0
# To disable the HDR mode, please type the following into a terminal window before starting Picamera2:
## v4l2-ctl --set-ctrl wide_dynamic_range=0 -d /dev/v4l-subdev0
# you want to know more about HDR mode, look at 
# <https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf>
@error_handler
class capture_detect:
    def __init__(self, 
                 save_dir, 
                 w_h, 
                 lsize=(320, 240), 
                 mse = int(7), 
                 WHITE_BALANCE=True, 
                 awbmode="Auto", 
                 ):
        picam = Picamera2()
        video_config = picam.create_video_configuration(main={"size": w_h, "format": "RGB888"}, lores={"size": lsize, "format": "YUV420"})
        picam.configure(video_config)

        if WHITE_BALANCE:
            picam.set_controls({"AwbEnable": True})
            if awbmode == "Auto":# any illumant
                picam.set_controls({"AwbMode": controls.AwbModeEnum.Auto})
            elif awbmode == "Tungsten":# tungsten lighting
                picam.set_controls({"AwbMode": controls.AwbModeEnum.Tungsten})
            if awbmode == "Flourescent":# fluorescent lighting
                picam.set_controls({"AwbMode": controls.AwbModeEnum.Flourescent})
            if awbmode == "Indoor":#  indoor illumination
                picam.set_controls({"AwbMode": controls.AwbModeEnum.Indoor})
            if awbmode == "Daylight":# daylight illumination
                picam.set_controls({"AwbMode": controls.AwbModeEnum.Daylight})
            if awbmode == "Cloudy":# cloudy illumination
                picam.set_controls({"AwbMode": controls.AwbModeEnum.Cloudy})
            if awbmode == "Custom":# custom setting
                picam.set_controls({"AwbMode": controls.AwbModeEnum.Custom})


        self.save_dir = save_dir
        self.lsize = lsize
        self.picam = picam
        self.mse = mse

    def motion_detect(self, analyze_q, key_wait_q):
        self.picam.start_preview()
        encoder = H264Encoder(1000000, repeat=True)
        encoder.output = CircularOutput()
        self.picam.start()
        self.picam.start_encoder(encoder)

        w, h = self.lsize
        prev = None
        encoding = False

        while True:
            try:
                _key_wait = key_wait_q.get_nowait()
                # 特定のキーが押されたら終了する
                if _key_wait == int(0):
                    print("撮影を終了します")
                    analyze_q.put(int(0))
                    return
                # 撮影中断時間になる、かつ、撮影中でないなら中断する
                elif _key_wait == int(1) and not encoding:
                    _key_wait = key_wait_q.get()
                    # 中断時間中でも、特定のキーが押されたら終了する
                    if _key_wait == int(0):
                        print("撮影を終了します")
                        analyze_q.put(int(0))
                        return
                    # 中断時間が終了したら再開する
                    elif _key_wait == int(1):
                        pass


            except queue.Empty:
                cur = self.picam.capture_buffer("lores")
                cur = cur[:w * h].reshape(h, w)
                if prev is not None:
                    # Measure pixels differences between current and
                    # previous frame
                    mse = np.square(np.subtract(cur, prev)).mean()
                    if mse > self.mse:
                        if not encoding:
                            now = datetime.datetime.now()
                            now_time = now.strftime("%Y_%m_%d_%H_%M_%S")
                            local_file_path = f"{self.save_dir}/{now_time}.h264"
                            encoder.output.fileoutput = local_file_path
                            encoder.output.start()
                            encoding = True
                            print("New Motion", mse)
                        ltime = time.time()
                    else:
                        if encoding and time.time() - ltime > 5.0:
                            encoder.output.stop()
                            
                            now_date = now.strftime("%Y_%m_%d")
                            now_hour = now.strftime("%H")
                            
                            analyze_q.put(now_date)
                            analyze_q.put(now_hour)
                            analyze_q.put(f"{now_time}.h264")
                            analyze_q.put(local_file_path)

                            encoding = False
                            
                prev = cur

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import csv
import platform
import torch

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


class detect:
    def __init__(
                self, 
                weights=".yolov5/yolov5s.pt",  # model path or triton URL
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
                project=".",  # save results to project/name
                name="result",  # save results to project/name
                exist_ok=True,  # existing project/name ok, do not increment
                line_thickness=3,  # bounding box thickness (pixels)
                hide_labels=False,  # hide labels
                hide_conf=False,  # hide confidences
                half=False,  # use FP16 half-precision inference
                dnn=False,  # use OpenCV DNN for ONNX inference
                vid_stride=1,  # video frame-rate stride) -> None:
                ):
        
        self.weights = weights
        self.data = data
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.view_img = view_img
        self.save_txt = save_txt
        self.save_csv = save_csv
        self.save_conf = save_conf
        self.save_crop = save_crop
        self.nosave = nosave
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.update = update
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half
        self.dnn = dnn
        self.vid_stride = vid_stride


    @smart_inference_mode()
    def detect(
            self, 
            analyze_q, 
            file_name_q, 
            ):
                
        while True:        
            captured_date = analyze_q.get()
            if captured_date == int(0):
                print("物体検出を終了します")
                file_name_q.put(int(0))
                return
            captured_hour = analyze_q.get()
            captured_file_name = analyze_q.get()
            local_file_path = analyze_q.get()

            source = str(local_file_path)
            save_img = not self.nosave and not source.endswith(".txt")  # save inference images
            is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
            is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
            webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
            screenshot = source.lower().startswith("screen")
            if is_url and is_file:
                source = check_file(source)  # download

            # Directories
            save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
            (save_dir / "labels" if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir


            # Load model
            device = select_device(self.device)
            model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
            stride, names, pt = model.stride, model.names, model.pt
            imgsz = check_img_size(self.imgsz, s=stride)  # check image size


            # Dataloader
            bs = 1  # batch_size
            if webcam:
                view_img = check_imshow(warn=True)
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=self.vid_stride)
                bs = len(dataset)
            elif screenshot:
                view_img = self.view_img ### added by shunkode
                dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
            else:
                view_img = self.view_img ### added by shunkode
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=self.vid_stride)
            vid_path, vid_writer = [None] * bs, [None] * bs

            # Run inference
            model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
            seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
            for path, im, im0s, vid_cap, s in dataset:
                with dt[0]:
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    if model.xml and im.shape[0] > 1:
                        ims = torch.chunk(im, im.shape[0], 0)

                # Inference
                with dt[1]:
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False
                    if model.xml and im.shape[0] > 1:
                        pred = None
                        for image in ims:
                            if pred is None:
                                pred = model(image, augment=self.augment, visualize=visualize).unsqueeze(0)
                            else:
                                pred = torch.cat((pred, model(image, augment=self.augment, visualize=visualize).unsqueeze(0)), dim=0)
                        pred = [pred, None]
                    else:
                        pred = model(im, augment=self.augment, visualize=visualize)
                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                # Define the path for the CSV file
                csv_path = save_dir / "predictions.csv"

                # Create or append to the CSV file
                def write_to_csv(image_name, prediction, confidence):
                    data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
                    with open(csv_path, mode="a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=data.keys())
                        if not csv_path.is_file():
                            writer.writeheader()
                        writer.writerow(data)

                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f"{i}: "
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # im.jpg
                    txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
                    s += "%gx%g " % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if self.save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            label = names[c] if self.hide_conf else f"{names[c]}"
                            confidence = float(conf)
                            confidence_str = f"{confidence:.2f}"

                            if self.save_csv:
                                write_to_csv(p.name, label, confidence_str)

                            if self.save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                                with open(f"{txt_path}.txt", "a") as f:
                                    f.write(("%g " * len(line)).rstrip() % line + "\n")

                            if save_img or self.save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if self.hide_labels else (names[c] if self.hide_conf else f"{names[c]} {conf:.2f}")
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            if self.save_crop:
                                save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                    # Stream results
                    im0 = annotator.result()
                    if view_img:
                        if platform.system() == "Linux" and p not in windows:
                            windows.append(p)
                            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == "image":
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if vid_path[i] != save_path:  # new video
                                vid_path[i] = save_path
                                if isinstance(vid_writer[i], cv2.VideoWriter):
                                    vid_writer[i].release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                            vid_writer[i].write(im0)

                # Print time (inference-only)
                LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

            # Print results
            t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
            LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
            if self.save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if self.save_txt else ""
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
            if self.update:
                strip_optimizer(self.weights[0])  # update model (to fix SourceChangeWarning)

            file_name_q.put(captured_date)
            file_name_q.put(captured_hour)
            file_name_q.put(captured_file_name)
            file_name_q.put(local_file_path)

            
            analyzed_file_name = f"{Path(captured_file_name).stem}.mp4"
            analyzed_file_path = f"{save_dir}/{analyzed_file_name}"

            file_name_q.put(captured_date)
            file_name_q.put(captured_hour)
            file_name_q.put(analyzed_file_name)
            file_name_q.put(analyzed_file_path)

@error_handler
class send_immediately:
    def __init__(self, 
                 dbx_folder_path, 
                 ):
        
        self.dbx_folder_path = dbx_folder_path

    def send_immediately(self, dbx, file_name_q):
        while True:
                max_retries = 3
                retry_delay = 60

                creation_day = file_name_q.get()
                if creation_day == int(0):
                    print("転送を終了します")
                    return
                creation_hour = file_name_q.get()
                file_name = file_name_q.get()
                local_file_path = file_name_q.get()
                dropbox_path = f"/{self.dbx_folder_path}/{creation_day}/{creation_hour}/{file_name}"
                try:
                    for attempt in range(max_retries + 1):
                        try:
                            #dropboxにアップロード
                            dbx.files_upload(open(local_file_path, 'rb').read(), dropbox_path)#Upload pathを設定する!
                            print(f"{file_name}転送完了")
                            # 送信済のファイルを消去
                            #os.remove(local_file_path)
                            break
                        except requests.exceptions.ConnectionError as e:
                            # エラー発生時刻, 関数名, エラー内容を記載
                            now = datetime.datetime.now()
                            now = now.strftime("%Y_%m_%d_%H_%M_%S")
                            with open ("log.txt", "a") as log:
                                log.write(f"エラー発生時刻: {now}, 関数名: {sys._getframe().f_code.co_name}, エラー内容: ({type(e).__name__}): {e}\n")
                            print(f"An error occurred: {e} ({type(e).__name__})")
                            print(f"Connection error (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds...")
                            # 指定時間待機
                            time.sleep(retry_delay)
                            retry_delay *= 2
                            # アクセストークンを更新する
                            dbx = update_access_token()
                    else:
                        print("Maximum number of retries reached. Connection could not be established.")
                        # 転送に失敗した場合、再びQueueに転送するファイルの情報を格納
                        file_name_q.put(creation_day)
                        file_name_q.put(creation_hour)
                        file_name_q.put(file_name)
                        file_name_q.put(local_file_path)

                except (dropbox.exceptions.AuthError, TimeoutError, requests.exceptions.ReadTimeout) as e:
                    print(e)
                    dbx = update_access_token()  
                    # 転送に失敗した場合、再びQueueに転送するファイルの情報を格納
                    file_name_q.put(creation_day)
                    file_name_q.put(creation_hour)
                    file_name_q.put(file_name)
                    file_name_q.put(local_file_path)

                except (TimeoutError) as e:
                    print(e)
                    print(f"Due to Timeout Error, I couldn't send {local_file_path}")
                    # 転送に失敗した場合、再びQueueに転送するファイルの情報を格納
                    file_name_q.put(creation_day)
                    file_name_q.put(creation_hour)
                    file_name_q.put(file_name)
                    file_name_q.put(local_file_path)
                    pass

                # Dropbox上に同名のファイルがあればエラーが起きるため、同名のファイルがあるか確認し、あればローカルのファイルを消去する
                except(dropbox.exceptions.ApiError) as e:
                    print(e)
                    try:
                        # Attempt to get the metadata of the file
                        file_metadata = dbx.files_get_metadata(dropbox_path)
                        print(f"同名のファイルが存在します。ローカルのファイルを消去した後、送信を再開します: {file_metadata.name}")
                        #os.remove(local_file_path)
                    # ファイルが存在する以外が原因のApiErrorの場合、エラーメッセージを出力
                    except dropbox.exceptions.ApiError as e:
                        if e.error.is_path() and e.error.get_path().is_conflict():
                            print("同一のファイルはありませんが、別のエラーが発生しています.")
                            # 転送に失敗した場合、再びQueueに転送するファイルの情報を格納
                            file_name_q.put(creation_day)
                            file_name_q.put(creation_hour)
                            file_name_q.put(file_name)
                            file_name_q.put(local_file_path)

                        else:
                            print(f"An error occurred: {e}")
                            # 転送に失敗した場合、再びQueueに転送するファイルの情報を格納
                            file_name_q.put(creation_day)
                            file_name_q.put(creation_hour)
                            file_name_q.put(file_name)
                            file_name_q.put(local_file_path)
      

@error_handler
def concurrent_execution(capture_detect,
                         interruption_time, 
                         restart_time, 
                         send_immediately, 
                         analyze=None):
    
    if not os.path.isdir(capture_detect.save_dir):
        print("指定したローカルディレクトリは存在しません")

    if analyze is not None:
        analyze_q = queue.Queue()
    # 撮影ファイルのローカルパスを渡すためのQueue
    file_name_q = queue.Queue()
    # エスケープキーが押下された際に、ある値を渡すためのQueue
    key_wait_q = queue.Queue()

    # Executor.ThreadPoolExecutorの実行用に、関数と引数のリストを作成する
    executor_func_list = []
    executor_args_list = []

    if analyze is not None:
        # 撮影用の関数, 引数リスト
        executor_func_list.append(capture_detect.motion_detect)
        executor_args_list.append([analyze_q, key_wait_q])
    elif analyze is None:
        executor_func_list.append(capture_detect.motion_detect)
        executor_args_list.append([file_name_q, key_wait_q])
    
    if analyze is not None:
        executor_func_list.append(analyze.detect)
        executor_args_list.append([analyze_q, file_name_q])
    elif analyze is None:
        print("物体検出は行いません")
        pass
    
    #for camera_number in camera_numbers_list:
    #    executor_func_list.append(Capture.save_frame)
    #    executor_args_list.append([camera_number])
    # データ送信用の関数、引数リスト
    if send_immediately:
        dbx = update_access_token()
        executor_func_list.append(send_immediately.send_immediately)
        executor_args_list.append([dbx, file_name_q])
    elif not send_immediately:
        print("\nデータ転送は行いません\n")


    # scheduled_timeの年月日を基準にしているため、再開時間が中断時間より前の場合は、1日分ずらす
    # 例えば、
    # scheduled_time が 2023/1/1 ... で、
    # interruption_time = 18:00
    # restart_time = 6:00 のとき、
    # 中断時間 = 2023/1/1 18:00
    # 再開時間 = 2023/1/1  6:00
    # となってしまい、矛盾が生じるため、この矛盾を解決するため再開時間を1日ずらす
    if restart_time < interruption_time:
        restart_time += 86400 #86400=24*60*60
    # UNIX時間（積算秒数）に直す
    scheduled_time = input("プログラム開始時間を入力してください")
    
    # interruption_timeは時刻で入力するため、scheduled_timeの年月日を基準にする
    _interruption_time_ = scheduled_time[0:8]
    _restart_time_ = _interruption_time_
    # interruption_timeとrestart_timeが、06:00のように5文字になっていなければ終了する
    if len(interruption_time) != 5:
        print("interruption_time does not match format")
        sys.exit()
    elif len(restart_time) != 5:
        print("restart_time does not match format")
        sys.exit()
    interruption_time = str(_interruption_time_) + str(interruption_time[0:2]) + str(interruption_time[3:5] + "00")
    restart_time = str(_restart_time_) + str(restart_time[0:2]) + str(restart_time[3:5] + "00")
    interruption_time = time.mktime(time.strptime(interruption_time, "%Y%m%d%H%M%S"))
    restart_time = time.mktime(time.strptime(restart_time, "%Y%m%d%H%M%S"))

    scheduled_time = time.mktime(time.strptime(scheduled_time, "%Y%m%d%H%M%S"))


    # 再開時間よりプログラム開始時間が遅ければ、
    # 中断時間、再開時間ともに1日ずらす
    # 例えば、
    # scheduled_time = 2023/1/1 19:00
    # interruption_time = 15:00
    # restart_time = 18:00
    # の場合、
    # 中断時間 = 2023/1/1 15:00
    # 再開時間 = 2023/1/1 18:00
    # となると中断が起きなくなってしまうため、1日ずらす
    if restart_time < scheduled_time:
        interruption_time += 86400 #86400=24*60*60
        restart_time += 86400 #86400=24*60*60

    
    # 既に中断する時間内であれば、中断するか撮影開始するか確認する。
    if interruption_time < scheduled_time < restart_time:
        y_n = input("次の再開時間まで中断します。よろしいですか？\nよろしければ\"y\"を、\nいますぐ撮影開始する場合は\"n\"を入力してください。")
        if y_n == "y":
            print("テストを行った後、再開時間まで中断します")


        elif y_n == "n":
            print("撮影開始")
            # 撮影続行する場合は、中断時間、再開時間ともに1日ずらす
            interruption_time += 86400 #86400=24*60*60
            restart_time += 86400 #86400=24*60*60


    # キーボード待機関数と引数を代入
    executor_func_list.append(keyboard_wait_repeat)
    executor_args_list.append([int(2), int(1), key_wait_q, interruption_time, restart_time])

    print(executor_func_list)
    print(executor_args_list)

    

    if ((scheduled_time - time.time()) >0):
        print("開始予定時間まで待機します")
        time.sleep(scheduled_time - time.time())


    # Create a ProcessPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Use zip to pair each task with its corresponding argument pair and submit to executor
        futures = [executor.submit(func, *args) for func, args in zip(executor_func_list, executor_args_list)]
        
        #concurrent.futures.wait(futures)
        

    print("終了")
    # Wait for all tasks to complete
    #concurrent.futures.wait(futures)
            
    # Check if any errors occurred and print exception information
    for future in futures:
        try:
            # Accessing the result will raise an exception if the task had an error
            result = future.result()
            print("result: ", result)
        except Exception as e:
            print(f"Error occurred: {e}")
    
