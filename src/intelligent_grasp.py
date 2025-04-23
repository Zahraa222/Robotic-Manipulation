#!/usr/bin/python3
# coding=utf8

import sys
import cv2
import time
import math
import rospy
import numpy as np
from threading import RLock, Timer, Thread

from std_srvs.srv import *
from std_msgs.msg import *
from sensor_msgs.msg import Image

from visual_processing.msg import Result
from visual_processing.srv import SetParam
from hiwonder_servo_msgs.msg import MultiRawIdPosDur
from ros_robot_controller.msg import RGBState, RGBsState, BuzzerState

from armpi_pro import pid
from armpi_pro import misc
from armpi_pro import bus_servo_control
from kinematics import ik_transform

lock = RLock()
ik = ik_transform.ArmIK()

x_dis = 500
y_dis = 0.15
img_w = 640
img_h = 480
centreX = 320
centreY = 410
offset_y = 0
stable = False
arm_move = False
__isRunning = False
position_en = False
detect_color = 'None'
x_pid = pid.PID(P=0.06, I=0, D=0)    # PID initialization
y_pid = pid.PID(P=0.00003, I=0, D=0)



range_rgb = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}

# Initial position
def initMove(delay=True):
    with lock:
        target = ik.setPitchRanges((0, 0.15, 0.03), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(joints_pub, 1.8, ((1, 200), (2, 500), (3, servo_data['servo3']), (4, servo_data['servo4']),
                                                                                (5, servo_data['servo5']),(6, servo_data['servo6'])))
    if delay:
        rospy.sleep(2)
        
# Set onboard RGB light
def set_rgb(r, g, b):
    rgb1 = RGBState()
    rgb1.id = 1
    rgb1.r = r
    rgb1.g = g
    rgb1.b = b
    rgb2 = RGBState()
    rgb2.id = 2
    rgb2.r = r
    rgb2.g = g
    rgb2.b = b
    msg = RGBsState()
    msg.data = [rgb1, rgb2]
    rgb_pub.publish(msg)

# Set buzzer    
def set_buzzer(freq, on_time, off_time, repeat):
    msg = BuzzerState()
    msg.freq = freq
    msg.on_time = on_time
    msg.off_time = off_time
    msg.repeat = repeat
    buzzer_pub.publish(msg)


# Reset variables
def reset():
    global arm_move
    global __isRunning
    global x_dis, y_dis
    global detect_color
    global position_en
    
    with lock:
        x_dis = 500
        y_dis = 0.15
        x_pid.clear()
        y_pid.clear()
        set_rgb(0, 0, 0)
        arm_move = False
        position_en = False
        detect_color = 'None'

# App initialization call
def init():
    rospy.loginfo("intelligent grasp Init")
    initMove()
    reset()
    
# Robot movement function
def move():
    global arm_move
    global detect_color
    
    coord_list = { 'red':(-0.2, 0.15, -0.06),
                   'green':(-0.2, 0.05, -0.06),
                   'blue':(-0.2, -0.05, -0.06)}
    
    while __isRunning:
        if arm_move and detect_color != 'None': # Wait until ready to grasp
            target_color = detect_color  # 暂存目标颜色
            set_rgb(range_rgb[target_color][2], range_rgb[target_color][1], range_rgb[target_color][0])    # Set RGB light color
            rospy.sleep(0.1)
            set_buzzer(1900, 0.1, 0.9, 1) # Make the buzzer beep once
            bus_servo_control.set_servos(joints_pub, 500, ((1, 120),)) # Open the robotic claw
            rospy.sleep(0.5)
            target = ik.setPitchRanges((0, round(y_dis + offset_y, 4), -0.08), -180, -180, 0) # Extend the robotic arm downward
            if target:
                servo_data = target[1]
                bus_servo_control.set_servos(joints_pub, 1, ((3, servo_data['servo3']), (4, servo_data['servo4']),
                                                                (5, servo_data['servo5']), (6, x_dis)))
            rospy.sleep(1.5)
            bus_servo_control.set_servos(joints_pub, 0.5, ((1, 450),)) # Close the robotic claw
            rospy.sleep(0.8)
            
            bus_servo_control.set_servos(joints_pub, 1.5, ((1, 450), (2, 500), (3, 80), (4, 825), (5, 625), (6, 500))) # Lift the robotic arm
            rospy.sleep(1.5)
            
            target = ik.setPitchRanges(coord_list[target_color], -180, -180, 0) # Move the robotic arm to place the color block
            if target:
                servo_data = target[1]
                bus_servo_control.set_servos(joints_pub, 1.2, ((6, servo_data['servo6']),)) # Rotate the robotic arm first
                rospy.sleep(1)
                bus_servo_control.set_servos(joints_pub, 1.5, ((3, servo_data['servo3']), (4, servo_data['servo4']), (5, servo_data['servo5']))) # Then lower it
            rospy.sleep(1.8)

            bus_servo_control.set_servos(joints_pub, 0.5, ((1, 150),))  # Open the robotic claw
            rospy.sleep(0.8)
            
            # Reset the robotic arm
            target = ik.setPitchRanges((0, 0.15, 0.03), -180, -180, 0)
            if target:
                servo_data = target[1]
                bus_servo_control.set_servos(joints_pub, 1, ((1, 200), (2, 500), (3, servo_data['servo3']),
                                                                (4, servo_data['servo4']), (5, servo_data['servo5'])))
                rospy.sleep(1)
                bus_servo_control.set_servos(joints_pub, 1.5, ((6, servo_data['servo6']),))
                rospy.sleep(1.5)
                
            reset()  # Reset variables
            
        else:
            rospy.sleep(0.01)

n = 0
num = 0
last_x = 0
last_y = 0
color_buf = []
color_list = {0:'None', 1:'red', 2:'green', 3:'blue'}
# Image processing result callback function
def run(msg):
    global arm_move
    global color_buf
    global offset_y
    global last_x, last_y
    global position_en, n
    global x_dis, y_dis, num
    global detect_color, stable
    
    # Update color block position parameters
    center_x = msg.center_x
    center_y = msg.center_y
    color_num = msg.data
    
    if not position_en: # Check if the color block is stably placed
        dx = abs(center_x - last_x)
        dy = abs(center_y - last_y)
        last_x = center_x
        last_y = center_y
        if dx < 3 and dy < 3:
            n += 1
            if n == 10:
                n = 0
                position_en = True # Stably placed
        else:
            n = 0
    
    else:
        # Color block is stable, start tracking and grasping
        if not arm_move and color_num != 0:
            diff_x = abs(center_x - centreX)
            diff_y = abs(center_y - centreY)
            # X-axis PID tracking
            if diff_x < 10:
                x_pid.SetPoint = center_x  # Setpoint
            else:
                x_pid.SetPoint = centreX
                
            x_pid.update(center_x)   # Current value
            dx = x_pid.output        # Output
            x_dis += int(dx)     
            x_dis = 200 if x_dis < 200 else x_dis
            x_dis = 800 if x_dis > 800 else x_dis
            # Y-axis PID tracking
            if diff_y < 10:
                y_pid.SetPoint = center_y  # Setpoint
            else:
                y_pid.SetPoint = centreY
                
            y_pid.update(center_y)   # Current value
            dy = y_pid.output        # Output
            y_dis += dy  
            y_dis = 0.12 if y_dis < 0.12 else y_dis
            y_dis = 0.28 if y_dis > 0.28 else y_dis
            
            # Move the robotic arm above the color block
            target = ik.setPitchRanges((0, round(y_dis, 4), 0.03), -180, -180, 0)
            if target:
                servo_data = target[1]
                bus_servo_control.set_servos(joints_pub, 20,((3, servo_data['servo3']),(4, servo_data['servo4']),
                                                             (5, servo_data['servo5']), (6, x_dis)))
            
            if dx < 2 and dy < 0.003 and not stable: # Wait for the robotic arm to stabilize above the color block
                num += 1
                if num == 10:
                    stable = True
                    num = 0
            else:
                num = 0
            
            if stable: # Confirm the detected color multiple times
                color_buf.append(color_num)
                if len(color_buf) == 5:
                    mean_num = np.mean(color_buf)
                    if mean_num == 1.0 or mean_num == 2.0 or mean_num == 3.0:
                        detect_color = color_list[mean_num]
                        offset_y = misc.map(target[2], -180, -150, -0.04, 0.03) # Set position offset
                        arm_move = True # Set robotic arm to grasp
                    color_buf = []
                    stable = False
                    

result_sub = None
heartbeat_timer = None
# Enter service callback function
def enter_func(msg):
    global lock
    global result_sub
    
    rospy.loginfo("enter intelligent grasp")
    init()
    with lock:
        if result_sub is None:
            rospy.ServiceProxy('/visual_processing/enter', Trigger)()
            result_sub = rospy.Subscriber('/visual_processing/result', Result, run)
            
    return [True, 'enter']

# Exit service callback function
def exit_func(msg):
    global lock
    global result_sub
    global __isRunning
    global heartbeat_timer
    
    rospy.loginfo("exit intelligent grasp")
    with lock:
        rospy.ServiceProxy('/visual_processing/exit', Trigger)()
        __isRunning = False
        reset()
        try:
            if result_sub is not None:
                result_sub.unregister()
                result_sub = None
            if heartbeat_timer is not None:
                heartbeat_timer.cancel()
                heartbeat_timer = None
        except BaseException as e:
            rospy.loginfo('%s', e)
        
    return [True, 'exit']

# Start running function
def start_running():
    global lock
    global __isRunning
    
    rospy.loginfo("start running intelligent grasp")
    with lock:
        __isRunning = True
        visual_running = rospy.ServiceProxy('/visual_processing/set_running', SetParam)
        visual_running('colors','rgb')
        rospy.sleep(0.1)
        # Start the sub-thread
        th = Thread(target=move)
        th.setDaemon(True)
        th.start()

# Stop running function
def stop_running():
    global lock
    global __isRunning
    
    rospy.loginfo("stop running intelligent grasp")
    with lock:
        reset()
        __isRunning = False
        initMove(delay=False)
        rospy.ServiceProxy('/visual_processing/set_running', SetParam)()
        
# Set running service callback function
def set_running(msg):
    if msg.data:
        start_running()
    else:
        stop_running()
        
    return [True, 'set_running']

# Heartbeat service callback function
def heartbeat_srv_cb(msg):
    global heartbeat_timer

    if isinstance(heartbeat_timer, Timer):
        heartbeat_timer.cancel()
    if msg.data:
        heartbeat_timer = Timer(5, rospy.ServiceProxy('/intelligent_grasp/exit', Trigger))
        heartbeat_timer.start()
    rsp = SetBoolResponse()
    rsp.success = msg.data

    return rsp


if __name__ == '__main__':
    # Initialize node
    rospy.init_node('intelligent_grasp', log_level=rospy.DEBUG)
    # Servo publishing
    joints_pub = rospy.Publisher('/servo_controllers/port_id_1/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)
    # App communication services
    enter_srv = rospy.Service('/intelligent_grasp/enter', Trigger, enter_func)
    exit_srv = rospy.Service('/intelligent_grasp/exit', Trigger, exit_func)
    running_srv = rospy.Service('/intelligent_grasp/set_running', SetBool, set_running)
    heartbeat_srv = rospy.Service('/intelligent_grasp/heartbeat', SetBool, heartbeat_srv_cb)
    # Buzzer
    buzzer_pub = rospy.Publisher('/ros_robot_controller/set_buzzer', BuzzerState, queue_size=1)
    # RGB light
    rgb_pub = rospy.Publisher('/ros_robot_controller/set_rgb', RGBsState, queue_size=1)
    rospy.sleep(0.5) # Delay required after publishing to take effect
    
    debug = False
    if debug:
        rospy.sleep(0.2)
        enter_func(1)
        start_running()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
