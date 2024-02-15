# What's this?
This can shoot videos every time the object caught in a picamera moves.
Shot videos are sent to dropbox.

# System overview
![システムワークフロー_v1](https://github.com/shunkode/MotionDetection_with_picamera2_dropbox/assets/106649051/592ef63f-d8fd-4917-8f41-222229891311)

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

