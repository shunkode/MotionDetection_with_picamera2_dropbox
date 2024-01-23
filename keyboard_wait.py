import time
from judge_os import if_Windows_or_Linux


if if_Windows_or_Linux():
    print("I guess you are using Windows")
    import keyboard

    def wait_keyboard_with_Windows(wait_time):
        wait_start_time = time.time()
        print("You should push escape key if you want to finish")
        while ((time.time() - wait_start_time) < wait_time):
            if keyboard.is_pressed("escape"):
                print("I COMPLETE THIS WORK!!! \nGOOD JOB FOR TODAY!!!")
                return False
        print("finish waiting keyboard")
        return True
            
            
        
            

elif not if_Windows_or_Linux():
    print("I guess you are using Linux")
    import evdev
    import select

    # target_string（対象とする文字列）の中に、search_string（探したい文字列）が含まれているか判定するプログラム
    def if_contains_string(search_string, target_string):
        search_string = search_string.lower()
        target_string = target_string.lower()

        if search_string in target_string:
            return True
        else:
            return False

    # 接続しているUSBデバイスのパスを探す & イベントを取得するデバイスとして指定するプログラム
    def search_and_use_usb_device(search_device_name):
        try:
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            for dev in devices:
                #print(dev.path, dev.name)#, dev.phys)
                if if_contains_string(search_device_name, dev.name):
                    device_path = dev.path
            #print("device_path: ", device_path)
            device = evdev.InputDevice(device_path)
        except UnboundLocalError:
            print("キーボードが接続されていないか、キーボードの端末名を間違えている可能性があります")
            device = None
        return device

    ## Usage(search_and_use_usb_device("device_name  ex:keyboard, mouse, etc..."))


    def wait_keyboard_with_Linux(wait_time):
        keyboard = search_and_use_usb_device("keyboard")
        if keyboard is None:
            return True
        elif keyboard is not None:
            print("You should push escape key if you want to finish")
            # Escapeキーのコードを取得
            esc_keycode = evdev.ecodes.KEY_ESC

            # 3秒間入力待ちする
            timeout = wait_time

            # イベント待機ループ
            while True:
                # イベントを取得
                r, w, x = select.select([keyboard], [], [], timeout)
                if keyboard in r:
                    events = keyboard.read()
                    for event in events:
                        # event.type == evdev.ecodes.EV_KEY: イベントがキーボード由来のものか確認（=キーボードが操作されたか確認）
                        # event.code == esc_keycode: エスケープキーが押されたか確認
                        # event.value == 1: キーボードが押下されたか、離上されたか確認（離上=0, 押下=1, ホールド=2）
                        if event.type == evdev.ecodes.EV_KEY and event.code == esc_keycode and event.value == 1:
                            print("Escape key pressed")
                            return False
                            #break
                else:
                    print("Timeout occurred")
                    return True
                    #break

if __name__ == "__main__":
    keyboard = search_and_use_usb_device("keyboard")
    wait_keyboard_with_Linux(keyboard)
    
"""
## Usage(if_contains_string, search_and_use_usb_device, wait_keyboard)
keyboard = search_and_use_usb_device("keyboard")
wait_keyboard(wait_time)
"""


# 特定の時間指定してスリープとキーボード操作受付を繰り返す
def keyboard_wait_particular_time(interval, keyboard_wait_time):
    # キーボード待機時間を設定（処理時間を考慮し、0.1秒の余分な時間を作る）
    time.sleep(interval)
    if if_Windows_or_Linux():
            waiting_keyboard = wait_keyboard_with_Windows(keyboard_wait_time)
    else:
        waiting_keyboard = wait_keyboard_with_Linux(keyboard_wait_time)

    if waiting_keyboard:
        return True
    elif not waiting_keyboard:
        return False

# 指定した時間、スリープとキーボード操作受付を繰り返す
# 特定のキー（本プログラムではエスケープ）が押されると、終了させたい関数の数だけkey_wait_q にint(0)を挿入
## interval: スリープ時間
## keyboard_wait_time: キーボード操作受付時間
## key_wait_q: Queue
def keyboard_wait_repeat(interval, keyboard_wait_time, key_wait_q, interruption_time, restart_time):
    waiting_keyboard = True
    while waiting_keyboard:
        waiting_keyboard = keyboard_wait_particular_time(interval, keyboard_wait_time)
        now = time.time()
        # 中断時間開始時間を超えると中断する
        if now > interruption_time:
            key_wait_q.put(int(1))
            interruption_time += 86400
        # 再開時間を超えると再開する
        elif now > restart_time and restart_time < interruption_time:
            restart_time += 86400
            key_wait_q.put(int(1))
    # 関数を終了させるため、int(0)を挿入
    key_wait_q.put(int(0))

if __name__ == "__main__":
    import queue
    q = queue.Queue()
    keyboard_wait_repeat(5, 5, q)
    if q == int(0):
        print("正常終了")
    else:
        print("異常終了")


def keyboard_wait(scheduled_time):
    # キーボード待機時間を設定（処理時間を考慮し、0.1秒の余分な時間を作る）
    keyboard_wait_time = scheduled_time - time.time() - float(0.1)
    if if_Windows_or_Linux():
            waiting_keyboard = wait_keyboard_with_Windows(keyboard_wait_time)
    else:
        waiting_keyboard = wait_keyboard_with_Linux(keyboard_wait_time)

    if waiting_keyboard:
        return True
    elif not waiting_keyboard:
        return False

    """
    interval = scheduled_time - time.time()
    if interval < 10:
        rest_time = interval * 7 / 10
        keyboard_wait_time = interval / 10
        print(f"WE WILL WAIT {keyboard_wait_time} seconds after {rest_time} SECONDS. \nIF YOU WANT TO FINISH SENDING, YOU SHOULD PUSH ESCAPE in that time. \nI WON'T REPEAT THIS WORK.")
        time.sleep(rest_time)
        if if_Windows_or_Linux():
            waiting_keyboard = wait_keyboard_with_Windows(keyboard_wait_time)
        if not if_Windows_or_Linux():
            waiting_keyboard = wait_keyboard_with_Linux(keyboard_wait_time)

    
    elif 10 <= interval < 30:
        rest_time = interval * 3 / 10
        keyboard_wait_time = interval / 10
        print(f"WE WILL WAIT {keyboard_wait_time} seconds after {rest_time} SECONDS. \nIF YOU WANT TO FINISH SENDING, YOU SHOULD PUSH ESCAPE in that time. \nI'll conduct this work twice.")
        for i in range(2):
            time.sleep(rest_time)
            if if_Windows_or_Linux():
                waiting_keyboard = wait_keyboard_with_Windows(keyboard_wait_time)
                if not waiting_keyboard:
                    break
                
            if not if_Windows_or_Linux():
                waiting_keyboard = wait_keyboard_with_Linux(keyboard_wait_time)
                if not waiting_keyboard:
                    break
            


    elif 30 <= interval < 60:
        rest_time = interval * 2 / 10
        keyboard_wait_time = interval / 10
        print(f"WE WILL WAIT {keyboard_wait_time} seconds after {rest_time} SECONDS. \nIF YOU WANT TO FINISH SENDING, YOU SHOULD PUSH ESCAPE in that time. \nI'll conduct this work third time.")
        for i in range(3):
            time.sleep(rest_time)
            if if_Windows_or_Linux():
                waiting_keyboard = wait_keyboard_with_Windows(keyboard_wait_time)
                if not waiting_keyboard:
                    break
                
            if not if_Windows_or_Linux():
                waiting_keyboard = wait_keyboard_with_Linux(keyboard_wait_time)
                if not waiting_keyboard:
                    break
    
    elif 60 <= interval < 200:
        rest_time = interval  / 10
        keyboard_wait_time = interval / 10
        print(f"WE WILL WAIT {keyboard_wait_time} seconds after {rest_time} SECONDS. \nIF YOU WANT TO FINISH SENDING, YOU SHOULD PUSH ESCAPE in that time. \nI'll conduct this work fourth time.")
        for i in range(4):
            time.sleep(rest_time)
            if if_Windows_or_Linux():
                waiting_keyboard = wait_keyboard_with_Windows(keyboard_wait_time)
                if not waiting_keyboard:
                    break
                
            if not if_Windows_or_Linux():
                waiting_keyboard = wait_keyboard_with_Linux(keyboard_wait_time)
                if not waiting_keyboard:
                    break


            

    elif 200 <= interval:
        rest_time = 40
        keyboard_wait_time = 10
        print(f"WE WILL WAIT {keyboard_wait_time} seconds after {rest_time} SECONDS. \nIF YOU WANT TO FINISH SENDING, YOU SHOULD PUSH ESCAPE in that time.")
        while (scheduled_time - time.time()) >= 60:
            time.sleep(rest_time)
            if if_Windows_or_Linux():
                waiting_keyboard = wait_keyboard_with_Windows(keyboard_wait_time)
                if not waiting_keyboard:
                    break
            if not if_Windows_or_Linux():
                waiting_keyboard = wait_keyboard_with_Linux(keyboard_wait_time)
                if not waiting_keyboard:
                    break

    
    
    if waiting_keyboard:
        return True
    if not waiting_keyboard:
        return False
    """
"""
import sys
def determine_scheduled_time(scheduled_time, interval, DAY, HOUR, MINUTE, SECOND):
    if (DAY and HOUR) or (DAY and MINUTE) or (DAY and SECOND) or (HOUR and MINUTE) or (HOUR and SECOND) or (MINUTE and SECOND):
        print("Only one of DAY, HOUR, MINUTE and SECOND should be True")
    if SECOND:
        print("I guess you input SECOND")

    if DAY:
        scheduled_time =scheduled_time + interval * 86400
        print("I guess you input DAY")
    elif HOUR:
        scheduled_time =scheduled_time + interval * 3600
        print("I guess you input HOUR")
    elif MINUTE:
        scheduled_time =scheduled_time + interval * 60
        print("I guess you input MINUTE")
    elif SECOND:
        scheduled_time =scheduled_time + interval
        print("I guess you input SECOND")
    else:
        print("NOT MATCH FORMAT!")
        sys.exit()
    return scheduled_time
"""
"""
## Usage
scheduled_time = scheduled_time = determine_scheduled_time(scheduled_time, interval, DAY, HOUR, MINUTE, SECOND)
"""
