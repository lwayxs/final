import os
import tkinter
from tkinter import *
from tkinter import filedialog

import cv2
from PIL import Image, ImageTk

import emotion_recognize

photo = 'none'


#视频推理函数
def cut_out_video(input_video_path,out_video_path):
    video_read_cap = cv2.VideoCapture(input_video_path)

    input_video_width = int(video_read_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_video_height = int(video_read_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_video_fps = int(video_read_cap.get(cv2.CAP_PROP_FPS))
    input_video_fourcc = int(cv2.VideoWriter_fourcc(*'mp4v'))

    max_size=max(input_video_width,input_video_height)
    if max_size>450:
        scale = 450/ (max_size * 1.0)
    else:
        scale=1

    out_video_width = input_video_width*scale
    out_video_height = input_video_height*scale

    out_video_size = (int(out_video_width), int(out_video_height))
    video_write_cap = cv2.VideoWriter(out_video_path, input_video_fourcc, input_video_fps, out_video_size)

    while video_read_cap.isOpened():
        result, frame = video_read_cap.read()
        if not result:
            break
        # 再resize到目标尺寸
        target = cv2.resize(frame, (int(out_video_width), int(out_video_height)))
        video_write_cap.write(target)
    return out_video_path
#尺寸转换
def scale_image(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv2.resize(image, (width, height))
#尺寸调整 放在视频尺寸过大 无法在窗口显示
def shrink_image(path, max_size):
    # 比较长度和宽度，取最大值
    image=cv2.imread(path)
    length = image.shape[0]
    if length < image.shape[1]:
        length = image.shape[1]
    if length < max_size:
        return path
    else:
        scale = max_size / (length * 1.0)
        image=scale_image(image, scale)
        cv2.imwrite(path,image)
        return path

def shrink_img(image, max_size):
    # 比较长度和宽度，取最大值

    length = image.shape[0]

    scale = max_size / (length * 1.0)
    image=scale_image(image, scale)
    return image

# "点击图片火焰识别"按钮按下的函数
def image_fire():
    global photo
    now_path = os.path.join(os.getcwd(), 'save_dir')
    si_path=os.path.join(now_path,'pic_emo.jpg')
    path_ = filedialog.askopenfilenames(initialdir=os.path.dirname(__file__))
    path = path_[0]  # path_为元组，将地址从元组中取出
    img_path = path
    save_path= emotion_recognize.image_recognize(img_path, si_path)

    img=shrink_image(save_path,450) #缩放
    img = Image.open(img)  # 打开图片
    photo = ImageTk.PhotoImage(img)  # 用PIL模块的PhotoImage打开
    Label(myWindow, image=photo).place(relx=0.33, rely=0.18)  # 放置图片的标签

# "点击视频火焰识别"按钮按下的函数
def video_fire():
    global photo
    now_path = os.path.join(os.getcwd(), 'save_dir')
    resize_path=os.path.join(now_path,'resize_vid_emo.mp4')
    sv_path=os.path.join(os.getcwd(), 'result_vid_emo.avi')
    path_ = filedialog.askopenfilenames(initialdir=os.path.dirname(__file__))
    path = path_[0]  # path_为元组，将地址从元组中取出
    bool = path.endswith(".mp4")
    if bool:
        v_path =cut_out_video(path, resize_path)
        done_path= emotion_recognize.video_recognize(v_path, sv_path)
        Label_movie = Label(myWindow)  # 创建一个Label，用来安置后面的视频的帧
        Label_movie.place(relx=0.3, rely=0.13)
        video = cv2.VideoCapture(done_path)  # 将视频读取，并将其创建为一个cv对象

        def video_stream():
            result, movieFrame = video.read()
            if result==True:
                movieFrame = cv2.cvtColor(movieFrame, cv2.COLOR_BGR2RGB)


                image_cv = Image.fromarray(movieFrame)  # 将帧的图片序列转化为图片对象
                image_tk = ImageTk.PhotoImage(image=image_cv)  # 将图片对象传入tkinter支持的对象
                Label_movie.config(image=image_tk)  # 将Label的图片参数修改为image_tk
                Label_movie.image = image_tk
                Label_movie.after(5, video_stream)
        video_stream()
    else:
        Label(myWindow, text='请输入.MP4文件~', font=("黑体", 40), width=20, height=2).place(x=330, y=200, anchor='nw')
def put_real():
    global photo
    path=0
    now_path = os.path.join(os.getcwd(), 'save_dir')
    si_path=os.path.join(now_path,'fire_video.mp4')
    emotion_recognize.video_recognize(path, si_path)


def put_real_smoke():
    global photo
    path = 0
    now_path = os.path.join(os.getcwd(), 'save_dir')
    si_path = os.path.join(now_path, 'fire_video.mp4')
    emotion_recognize.smoke_recognize(path, si_path)


myWindow = Tk()
myWindow.title('疲劳状态异常检测')
myWindow.resizable(width=True, height=True)
width = 800
height = 500
screenwidth = myWindow.winfo_screenwidth()
screenheight = myWindow.winfo_screenheight()
alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
myWindow.geometry(alignstr)

put_title = tkinter.Button(myWindow, text='疲劳检测', font=('Arial', 10), width=13, height=1,command=put_real)
put_title.place(relx=0.42, rely=0.02)
exit_re = tkinter.Button(myWindow, text="退出", font=('Arial', 10), width=10, height=1, command=myWindow.destroy)
exit_re.place(relx=0.44, rely=0.9)
myWindow.mainloop()