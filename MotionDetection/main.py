from update_access_token import update_access_token
from Capture_and_Send import capture_detect, send_immediately, concurrent_execution
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
    Capture_data = capture_detect(save_dir="../../../1_data_raw", 
                                  w_h=(int(1280), int(720)), 
                                  lsize=(int(320), int(240)), 
                                  mse=int(7), 
                                  AUTO_WHITE_BALANCE=True, 
                                  awbmode="Auto"
                                  )

    # 撮影完了後すぐに転送したい場合は、True
    # データ転送を行わず、撮影のみを行う場合は、False
    SEND_IMMEDIATELY = True

    if SEND_IMMEDIATELY:
        Send_data = send_immediately(dbx_folder_path="Kodera/1_data_raw/TEST")
    elif not SEND_IMMEDIATELY:
        Send_data = None

    
    concurrent_execution(Capture_data, 
                         interruption_time="16:38", 
                         restart_time="16:40", 
                         send_immediately=Send_data)
