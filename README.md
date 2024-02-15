# What's this?
This can shoot videos every time the object caught in a picamera moves.  
Shot videos are sent to dropbox.  

The programs I made is following: main.py, Capture_and_Send.py, update_access_token.py, decorator.py, keyboard_wait.py, judge_os.py and other(README.md etc...).

# System overview
![system overview](https://github.com/shunkode/MotionDetection_with_picamera2_dropbox/assets/106649051/a6c2814f-4faf-4c20-8cc0-07823a67d5e1)

# Installation
```
git clone https://github.com/shunkode/MotionDetection_with_picamera2_dropbox.git
```
If you want to create python virtual environments, please input the following command.  
```
python -m venv --system-site-packages my-env
```
please assign the environment name to my-env.  
If you want to know more, please see the official manual.  
<https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf>  
  
Next, move to the MotionDetection directory and install library
```
cd MotionDetection
pip install -r requirements.txt
```
# Usage
Before you start this, you must add oauth2_refresh_token, app_key and app_secret to update_access_token.  
To start this program, execute the following command.
```
python main.py
```
main.py contains the changeable parameter like size, the destination local and dropbox directory, etc... .  
When you want to change these, please rewrite main.py  

