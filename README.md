# What's this?
This can shoot videos every time the object caught in a picamera moves.
Shot videos are sent to dropbox.

# Installation
```
git clone https://github.com/shunkode/MotionDetection_with_picamera2_dropbox.git
```
Move to the MotionDetection directory and install library
```
cd MotionDetection
pip install -r requirements.txt
```
# Usage
Before you start this, you must add oauth2_refresh_token, app_key and app_secret to update_access_token.  
In addition, to use this execute the following command.
```
python main.py
```
main.py contains the changeable parameter like size, the destination local and dropbox directory, etc... .  
When you want to change these, please rewrite main.py  

