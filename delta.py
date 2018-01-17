import imutils
import cv2
import time
import code
import numpy as np
import sys
import os
import subprocess

##############################################################################$
### Params
##############################################################################$
sensitivity = 55 # out of 100, higher is more sensitive
fps = 15
##############################################################################$
nz_threshold = 100 - sensitivity
consec_threshold = max(int((100 - sensitivity) / 10) - 1, 2)
events_threshold = max(int(consec_threshold / 2), 2)
frame_wait = 1.0 / fps
print("nz={} consec={} events={} wait={}".format(
    nz_threshold,
    consec_threshold,
    events_threshold,
    frame_wait
))

##############################################################################$
### Turn off screen while running ###
##############################################################################$
brightness = 1

def dark():
    global brightness
    out = subprocess.check_output(['brightness', '-l'])
    out = out.decode('utf-8')
    brightness = float(out.split('\n')[1].split(' ')[3])
    os.system('brightness 0')

def restore():
    os.system('brightness ' + str(brightness))

def exit(code):
    restore()
    sys.exit(code)

def alert():
    subprocess.check_call(
            ['play', 'alert.wav'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
    )
##############################################################################$

##############################################################################$
### Image processing helpers
##############################################################################$
def normalize(f):
    f = imutils.resize(f, width=500)
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return gray

def pnz(px):
    return int((np.count_nonzero(px) / float(px.size)) * 100)
##############################################################################$

##############################################################################$
### Main
##############################################################################$

# Gain access to camera
camera = cv2.VideoCapture(0)

# Wait for camera to load
time.sleep(0.5)

# Turn off screen
dark()

# Wait for lighting to adjust
time.sleep(0.5)

# Grab initial background image and preprocess
(grabbed, initial) = camera.read()
if not grabbed:
    print("Oh no! Couldn't grab first frame")
    exit(-1)
gray1 = normalize(initial)


moved_in_a_row = 0
events = 0
while True:

    # Grab current image
    time.sleep(frame_wait)
    (grabbed, frame2) = camera.read()
    if not grabbed:
        print("Oh no! Couldn't grab next frame")
        exit(-1)

    # Calculate clean B/W delta between current and original
    gray2 = normalize(frame2)
    delta = cv2.absdiff(gray1, gray2)
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
    dilated = cv2.dilate(thresh, None, iterations=2)


    # Number of changed pixels around edges
    top_nz = pnz(dilated[0])
    bottom_nz = pnz(dilated[-1])
    left_nz = pnz(dilated[:,0])
    right_nz = pnz(dilated[:,-1])
    # Total number of changed pixels
    total = pnz(dilated)

    # Check if this frame seems to indicate movement
    vert = top_nz > nz_threshold or bottom_nz > nz_threshold
    horz = left_nz > nz_threshold or right_nz > nz_threshold
    moved_this_time = False
    if vert and horz:
        moved_this_time = True
    elif (vert and total > nz_threshold) or (horz and total > nz_threshold):
        moved_this_time = True

    # Number of consecutive frames that indicate movement
    if moved_this_time:
        moved_in_a_row += 1
    else:
        moved_in_a_row = 0

    # If a lot of consecutive frames indicate movement, 
    # take a new background snapshot. 
    # If there is still movement after that, it's likely that the machine has actually moved.
    # If not, then it maybe it was just bumped
    if moved_in_a_row >= consec_threshold:
        events += 1
        moved_in_a_row = 0
        gray1 = gray2

    if events >= events_threshold:
        alert()
        exit(0)

    print("t={:03d} b={:03d} l={:03d} r={:03d} tot={:03d} {} {} {}".format(
        top_nz,
        bottom_nz,
        left_nz,
        right_nz,
        total,
        "!!!" if moved_this_time else "",
        moved_in_a_row,
        events
    ))

    cv2.imshow('dilated', dilated)
    cv2.waitKey(3)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #code.interact(local=locals())

