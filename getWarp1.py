import socket
import pickle
import struct
import cv2
import time
import os
import numpy as np
#import requests
dick = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
with open('param.txt') as f:
    K = eval(f.readline())
    D = eval(f.readline())

def undistort(img):
    DIM = img.shape[:2][::-1]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img[::]


def get_image():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = '192.168.0.11'
    port = 9988
    client_socket.connect((host_ip, port))

    data = b""
    payload_size = struct.calcsize("Q")
    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4*1024)
            if not packet: break
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4*1024)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)
        client_socket.close()
        return frame
numba=0
save_h = np.array([])
use=0
while True:
    frame = get_image()
    frame = undistort(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray, dick)
    c=[]
    if res[1] is not None and len(res[1])==4 and np.sum(res[1])==6:
        for i in range(4):
            marker=i
            index=np.where(res[1]==marker)[0][0]
            pt0=res[0][index][0][marker].astype(np.int16)
            c.append(list(pt0))
            cv2.circle(frame, pt0, 10, (0,0,255), thickness=-1)
        h, w, _ = frame.shape
        input_pt=np.array(c)
        output_pt=np.array([[0,0], [w,0],[w,h],[0,h]])
        h_, _ = cv2.findHomography(input_pt, output_pt)
        key = cv2.waitKey(ord('b')) & 0xFF
        if key == ord('b'):
            save_h= h_
            use = 1
    try:
        if use == 1:
            res_img=cv2.warpPerspective(frame, save_h, (w,h))
        else:
            res_img =cv2.warpPerspective(frame, h_, (w,h))
        cv2.imshow('pedophile',res_img)
    except:
        print("fuck you bitch!")
            

    
    cv2.imshow("img", frame)
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
