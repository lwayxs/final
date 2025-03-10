import os
import cv2
import ttkbootstrap as ttk
from ttkbootstrap import Style
from tkinter import filedialog
from PIL import Image, ImageTk
import emotion_recognize
import head_pose as hp  # 导入头部姿态检测模块
import threading

photo = 'none'

style = Style(theme="flatly")  
myWindow = style.master
myWindow.title('Fatigue state abnormal detection')
myWindow.geometry('800x500')
myWindow.resizable(width=False, height=False)

screenwidth = myWindow.winfo_screenwidth()
screenheight = myWindow.winfo_screenheight()
alignstr = '%dx%d+%d+%d' % (800, 500, (screenwidth - 800) / 2, (screenheight - 500) / 2)
myWindow.geometry(alignstr)



# -----------------------------------
def cut_out_video(input_video_path, out_video_path):
    video_read_cap = cv2.VideoCapture(input_video_path)
    input_video_width = int(video_read_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_video_height = int(video_read_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_video_fps = int(video_read_cap.get(cv2.CAP_PROP_FPS))
    input_video_fourcc = int(cv2.VideoWriter_fourcc(*'mp4v'))

    max_size = max(input_video_width, input_video_height)
    scale = 450 / max_size if max_size > 450 else 1

    out_video_width, out_video_height = int(input_video_width * scale), int(input_video_height * scale)
    video_write_cap = cv2.VideoWriter(out_video_path, input_video_fourcc, input_video_fps, (out_video_width, out_video_height))

    while video_read_cap.isOpened():
        result, frame = video_read_cap.read()
        if not result:
            break
        target = cv2.resize(frame, (out_video_width, out_video_height))
        video_write_cap.write(target)
    return out_video_path

def scale_image(image, scale):
    width, height = int(image.shape[1] * scale), int(image.shape[0] * scale)
    return cv2.resize(image, (width, height))

def shrink_image(path, max_size):
    image = cv2.imread(path)
    length = max(image.shape[0], image.shape[1])
    if length < max_size:
        return path
    scale = max_size / length
    image = scale_image(image, scale)
    cv2.imwrite(path, image)
    return path

# -----------------------------------

def image_fire():
    global photo
    now_path = os.path.join(os.getcwd(), 'save_dir')
    si_path = os.path.join(now_path, 'pic_emo.jpg')
    path_ = filedialog.askopenfilenames(initialdir=os.path.dirname(__file__))
    img_path = path_[0]
    save_path = emotion_recognize.image_recognize(img_path, si_path)

    img = shrink_image(save_path, 450)
    img = Image.open(img)
    photo = ImageTk.PhotoImage(img)
    ttk.Label(myWindow, image=photo).place(relx=0.33, rely=0.18)

def video_fire():
    global photo
    now_path = os.path.join(os.getcwd(), 'save_dir')
    resize_path = os.path.join(now_path, 'resize_vid_emo.mp4')
    sv_path = os.path.join(os.getcwd(), 'result_vid_emo.avi')
    path_ = filedialog.askopenfilenames(initialdir=os.path.dirname(__file__))
    path = path_[0]

    if path.endswith(".mp4"):
        v_path = cut_out_video(path, resize_path)
        # emotion_recognize.video_recognize 返回处理后的视频路径
        done_path = emotion_recognize.video_recognize(v_path, sv_path)
        Label_movie = ttk.Label(myWindow)
        Label_movie.place(relx=0.3, rely=0.13)

        # 新增：创建一个 Label 用于显示头部姿态检测结果
        head_pose_label = ttk.Label(myWindow, text="Head Pose: N/A", font=("Arial", 14))
        head_pose_label.place(relx=0.5, rely=0.05, anchor='center')

        video = cv2.VideoCapture(done_path)

        def video_stream():
            result, frame = video.read()
            if result:
                # 对当前帧调用头部姿态检测（注意：检测函数需返回 (pitch, yaw, roll) 或 None）
                pose = hp.estimate_head_pose(frame)
                print(pose)  # 检查这里的输出
                if pose is not None:
                    pitch, yaw, roll = pose
                    head_pose_label.config(text=f"Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}")
                else:
                    head_pose_label.config(text="No face detected")
                
                displayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_cv = Image.fromarray(displayFrame)
                image_tk = ImageTk.PhotoImage(image=image_cv)
                Label_movie.config(image=image_tk)
                Label_movie.image = image_tk
                Label_movie.after(5, video_stream)
            else:
                video.release()
        video_stream()
    else:
        ttk.Label(myWindow, text='请输入 .MP4 文件', font=("Arial", 14), bootstyle="danger").place(x=330, y=200)

def put_real():
    global photo
    path = 0
    now_path = os.path.join(os.getcwd(), 'save_dir')
    si_path = os.path.join(now_path, 'fire_video.mp4')
    emotion_recognize.video_recognize(path, si_path)

def put_real_smoke():
    global photo
    path = 0
    now_path = os.path.join(os.getcwd(), 'save_dir')
    si_path = os.path.join(now_path, 'fire_video.mp4')
    emotion_recognize.smoke_recognize(path, si_path)

dark_mode = False

def toggle_theme():
    global dark_mode
    if dark_mode:
        style.theme_use("flatly")  # 浅色模式
        toggle_button.config(text="🌙 Night pattern", bootstyle="secondary-outline")
    else:
        style.theme_use("darkly")  # 深色模式
        toggle_button.config(text="☀️ Daytime pattern", bootstyle="info-outline")
    dark_mode = not dark_mode

ttk.Label(myWindow, text="Fatigue state abnormal detection", font=("Helvetica", 18, "bold"), bootstyle="inverse-light").place(relx=0.5, rely=0.1, anchor='center')
ttk.Label(myWindow, text="Fatigue state abnormal detection", 
          font=("Helvetica", 18, "bold"), bootstyle="inverse-light"
         ).place(relx=0.5, rely=0.1, anchor='center')

# 新增按钮，调用video_fire
##ttk.Button(myWindow, text='🎥 Video detection with Head Pose', 
  ##         bootstyle="primary-outline", 
    ##       command=video_fire, padding=10).place(relx=0.5, rely=0.3, anchor='center')

# 其余已有按钮
ttk.Button(myWindow, text='😴 Fatigue detection', 
           bootstyle="success-outline", command=lambda: emotion_recognize.put_real_combined(), padding=25
          ).place(relx=0.5, rely=0.5, anchor='center')
ttk.Button(myWindow, text='⚠️ Anomaly detection', 
           bootstyle="warning-outline", command=put_real_smoke, padding=8
          ).place(relx=0.6, rely=0.5)
toggle_button = ttk.Button(myWindow, text="🌙 Night pattern", bootstyle="secondary-outline", command=toggle_theme, padding=5)
toggle_button.place(relx=0.5, rely=0.7, anchor='center')

ttk.Button(myWindow, text="❌ quit", bootstyle="danger-outline", command=myWindow.destroy, padding=5).place(relx=0.5, rely=0.85, anchor='center')

myWindow.mainloop()