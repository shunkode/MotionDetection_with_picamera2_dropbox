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

    def motion_detect(self, file_name_q, key_wait_q):
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
                    file_name_q.put(int(0))
                    return
                # 撮影中断時間になる、かつ、撮影中でないなら中断する
                elif _key_wait == int(1) and not encoding:
                    _key_wait = key_wait_q.get()
                    # 中断時間中でも、特定のキーが押されたら終了する
                    if _key_wait == int(0):
                        print("撮影を終了します")
                        file_name_q.put(int(0))
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
                            
                            file_name_q.put(now_date)
                            file_name_q.put(now_hour)
                            file_name_q.put(f"{now_time}.h264")
                            file_name_q.put(local_file_path)

                            encoding = False
                            
                prev = cur

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
                            os.remove(local_file_path)
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
                        os.remove(local_file_path)
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
def concurrent_execution(capture_detect, interruption_time, restart_time, send_immediately):
    if not os.path.isdir(capture_detect.save_dir):
        print("指定したローカルディレクトリは存在しません")

    
    # 撮影ファイルのローカルパスを渡すためのQueue
    file_name_q = queue.Queue()
    # エスケープキーが押下された際に、ある値を渡すためのQueue
    key_wait_q = queue.Queue()

    # Executor.ThreadPoolExecutorの実行用に、関数と引数のリストを作成する
    executor_func_list = []
    executor_args_list = []

    # 撮影用の関数, 引数リスト
    executor_func_list.append(capture_detect.motion_detect)
    executor_args_list.append([file_name_q, key_wait_q])
    
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
    executor_args_list.append([int(20), int(10), key_wait_q, interruption_time, restart_time])

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
    
