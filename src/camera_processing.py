#!/usr/bin/python3
# coding=utf8

import sys
import cv2
import time
import math
import rospy
import numpy as np
from armpi_pro import misc
from armpi_pro import apriltag
from threading import RLock
from std_srvs.srv import *
from std_msgs.msg import *
from sensor_msgs.msg import Image
from visual_processing.msg import Result
from visual_processing.srv import SetParam


lock = RLock()

image_sub = None
publish_en = True
size_s = (160, 120)
size_m = (320, 240)
__isRunning = False
target_type = 'None'
target_color = 'None'
id_smallest = 'None'
color_range_list = None
pub_time = time.time()

range_rgb = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}


# Find the contour with the largest area
# The parameter is the list of contours to compare
def getAreaMaxContour(contours):
    contour_area_temp = 0
    contour_area_max = 0
    area_max_contour = None
    for c in contours:  # Iterate over all contours
        contour_area_temp = math.fabs(cv2.contourArea(c))  # Calculate the area of the contour
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp > 10:  # Only consider it valid if area > 300, to filter out noise
                area_max_contour = c
    return area_max_contour, contour_area_max  # Return the largest contour

# Face detection function

# AprilTag detection function
detector = apriltag.Detector(searchpath=apriltag._get_demo_searchpath())
# Single color detection function
def color_detect(img, color):
    global pub_time
    global publish_en
    global color_range_list
    
    if color == 'None':
        return img
    
    msg = Result()
    area_max = 0
    area_max_contour = 0
    img_copy = img.copy()
    img_h, img_w = img.shape[:2]
    frame_resize = cv2.resize(img_copy, size_m, interpolation=cv2.INTER_NEAREST)
    frame_lab = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2LAB)  # Convert image to LAB color space

    if color in color_range_list:
        color_range = color_range_list[color]
        frame_mask = cv2.inRange(frame_lab, tuple(color_range['min']), tuple(color_range['max']))  # Bitwise operation on image and mask
        eroded = cv2.erode(frame_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))          # Erosion
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))            # Dilation
        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]         # Find contours
        area_max_contour, area_max = getAreaMaxContour(contours)                                   # Find the largest contour

        if area_max > 100:  # Found the largest area
            (centerx, centery), radius = cv2.minEnclosingCircle(area_max_contour)  # Get the minimum enclosing circle
            msg.center_x = int(misc.map(centerx, 0, size_m[0], 0, img_w))
            msg.center_y = int(misc.map(centery, 0, size_m[1], 0, img_h))
            msg.data = int(misc.map(radius, 0, size_m[0], 0, img_w))
            cv2.circle(img, (msg.center_x, msg.center_y), msg.data+5, range_rgb[color], 2)
            publish_en = True
        
        if publish_en:
            if (time.time()-pub_time) >= 0.06:
                result_pub.publish(msg)  # Publish the result
                pub_time = time.time()
                
            if msg.data == 0:
                publish_en = False
                result_pub.publish(msg)
        
    return img

# Multi-color detection function
# Camera image node callback function

def image_callback(ros_image):
    global lock
    global target_color

    # Convert ROS image message to OpenCV image
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    frame_result = cv2_img
    with lock:
        if __isRunning and target_color != 'None':
            frame_result = color_detect(cv2_img, target_color)

    rgb_image = cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB).tobytes()
    ros_image.data = rgb_image
    image_pub.publish(ros_image)



# Initialization call
def init():
    global publish_en
    global id_smallest
    global target_color
    global color_range_list
    
    publish_en = True
    id_smallest = 'None'
    target_color = 'None'
    rospy.loginfo("visual processing Init")
    color_range_list = rospy.get_param('/lab_config_manager/color_range_list', {})  # get lab range from ros param server

image_sub_st = False
def enter_func(msg):
    global lock
    global image_sub
    global image_sub_st
    
    rospy.loginfo("enter visual processing")
    init()
    with lock:
        if not image_sub_st:
            image_sub_st = True
            image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)
            
    return [True, 'enter']

def exit_func(msg):
    global lock
    global image_sub
    global image_sub_st
    global __isRunning
    global target_color
    
    rospy.loginfo("exit visual processing")
    with lock:
        __isRunning = False
        target_color = 'None'
        try:
            if image_sub_st:
                image_sub_st = False
                image_sub.unregister()
        except BaseException as e:
            rospy.loginfo('%s', e)
        
    return [True, 'exit']

def start_running():
    global lock
    global __isRunning
    
    rospy.loginfo("start running visual processing")
    with lock:
        __isRunning = True

def stop_running():
    global lock
    global __isRunning
    
    rospy.loginfo("stop running visual processing")
    with lock:
        __isRunning = False

def set_running(msg):
    global target_type 
    global target_color
    
    rospy.loginfo("%s", msg)
    init()
    if msg.type:
        target_type = msg.type
        target_color = msg.color
        start_running()
    else:
        target_color = 'None'
        stop_running()
        
    return [True, 'set_running']


if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('visual_processing', log_level=rospy.DEBUG)
    # Communication services
    image_pub = rospy.Publisher('/visual_processing/image_result', Image, queue_size=1)
    result_pub = rospy.Publisher('/visual_processing/result', Result, queue_size=1) 
    enter_srv = rospy.Service('/visual_processing/enter', Trigger, enter_func)
    exit_srv = rospy.Service('/visual_processing/exit', Trigger, exit_func)
    running_srv = rospy.Service('/visual_processing/set_running', SetParam, set_running)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    finally:
        cv2.destroyAllWindows()
