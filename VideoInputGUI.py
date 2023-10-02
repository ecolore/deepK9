# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:56:47 2023

@author: User
"""

import cv2
import tkinter as tk
from tkinter import filedialog
from topic import zoo_std


# 设置字体和其他显示参数
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 255, 0)  # 文本颜色 (B, G, R)
font_thickness = 2


dic_state = {0:"Lay",1:"Sit",2:"Stand",3:"Walk"}


# 创建一个 Tkinter 窗口
root = tk.Tk()
root.title("test_pose")

# 创建一个标签用于显示视频
label = tk.Label(root)
label.pack()

# 打开文件对话框以选择视频文件
def open_file():
    
    file_path = filedialog.askopenfilename(filetypes=[("视频文件", "*.mp4")])

    y = zoo_std.auto_std(file_path)

    if file_path:
        play_video(file_path,y)

# 播放视频
def play_video(file_path,y_predicted):
    cap = cv2.VideoCapture(file_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        

        
        current_text = dic_state[y_predicted]    
            

        # 在帧上叠加当前文本
        cv2.putText(frame, current_text, (50, 50), font, font_scale, font_color, font_thickness)

        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = tk.PhotoImage(data=cv2.imencode(".ppm", frame_rgb)[1].tobytes())

        label.config(image=img)
        label.image = img
        
        
        root.update()

    cap.release()
    cv2.destroyAllWindows()

# 创建打开文件按钮
open_button = tk.Button(root, text="上傳影片", command=open_file)
open_button.pack()

# 创建退出按钮
quit_button = tk.Button(root, text="退出", command=root.destroy)
quit_button.pack()

root.mainloop()
