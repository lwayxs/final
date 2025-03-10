
import cv2
import os
from onnx_infer import infer,smoke_infer
# emotion_recognize.py
from head_pose import estimate_head_pose, estimate_head_pose_frame


def video_recognize(video_path, model):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLOv8推理
        results = model.predict(frame)

        # 调用头部姿态估计
        head_pose = estimate_head_pose(frame)
        
        if head_pose:
            pitch, yaw, roll = head_pose
            if abs(yaw) > 45 or abs(pitch) > 30:
                print("头部角度偏移较大，请调整姿势！")
            else:
                print(f"头部姿态: pitch={pitch:.1f}, yaw={yaw:.1f}, roll={roll:.1f}")

        # 后续逻辑...


#屌用摄像头函数 #use-camera
def video_recognize(v_path, s_path):
    cap = cv2.VideoCapture(v_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # w
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # h
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # encode
    videoWriter = cv2.VideoWriter(s_path, fourcc, fps, (w, h))  #creat video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # video_fps

    if v_path == 0:
     while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = infer(frame)
        
        # 调用头部姿态检测
        pose = estimate_head_pose_frame(frame)
        if pose is not None:
            pitch, yaw, roll = pose
            cv2.putText(result, f"Pitch: {pitch:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(result, f"Yaw: {yaw:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(result, f"Roll: {roll:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(result, "No face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.imshow("Frame", result)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    else:
        while 1:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            if frame is None:
                break
            now_path = os.getcwd()
            result = infer(frame)
            #result = frame


            videoWriter.write(result)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    return s_path



#屌用摄像头函数 #use-camera
def smoke_recognize(v_path, s_path):
    cap = cv2.VideoCapture(v_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # w
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # h
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # encode
    videoWriter = cv2.VideoWriter(s_path, fourcc, fps, (w, h))  #creat video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # video_fps

    if v_path == 0:
        while 1:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            #frame = cv2.flip(frame, 1)
            #result = frame
            result = smoke_infer(frame)

            # show the output frame
            cv2.imshow("Frame", result)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
    else:
        while 1:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            if frame is None:
                break
            now_path = os.getcwd()
            result = smoke_infer(frame)
            #result = frame


            videoWriter.write(result)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    return s_path


def put_real_smoke(v_path, s_path):
    cap = cv2.VideoCapture(v_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # w
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # h
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # encode
    videoWriter = cv2.VideoWriter(s_path, fourcc, fps, (w, h))  #creat video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # video_fps

    if v_path == 0:
        while 1:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            #frame = cv2.flip(frame, 1)
            #result = frame
            result = infer(frame)

            # show the output frame
            cv2.imshow("Frame", result)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
    else:
        while 1:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            if frame is None:
                break
            now_path = os.getcwd()
            result = infer(frame)
            #result = frame


            videoWriter.write(result)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    return s_path

def put_real_combined(v_path=0, s_path=None):
    import numpy as np
    import head_pose as hp  # 确保引用了新增的 calculate_mar 函数
    cap = cv2.VideoCapture(v_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if s_path is None:
        now_path = os.getcwd()
        s_path = os.path.join(now_path, 'combined_output.avi')
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter(s_path, fourcc, fps, (w, h))

    # 滑动窗口参数
    window_size = 30
    fatigue_threshold = int(window_size * 0.7)
    fatigue_count = 0

    # 定义“Back”按钮区域（右上角）
    button_top_left = (w - 110, 10)
    button_bottom_right = (w - 10, 40)

    # 鼠标回调：点击“Back”按钮直接退出检测
    def mouse_callback(event, x, y, flags, param):
        nonlocal fatigue_count
        if event == cv2.EVENT_LBUTTONDOWN:
            if button_top_left[0] <= x <= button_bottom_right[0] and button_top_left[1] <= y <= button_bottom_right[1]:
                fatigue_count = -100  # 触发退出

    cv2.namedWindow("Combined Detection")
    cv2.setMouseCallback("Combined Detection", mouse_callback)

    # 新的疲劳判断函数：基于嘴部张开程度（MAR）和头部姿态
    def check_fatigue(frame, pitch, yaw):
        mar = hp.calculate_mar(frame)
        if mar is None:
            return False
        # 默认阈值
        threshold = 0.5
        # 当抬头时（pitch 较大），要求 MAR 更大才能认为疲劳
        if pitch is not None:
            if pitch > 10:   # 抬头
                threshold += 0.2
            elif pitch < -10:  # 低头时，门限降低
                threshold -= 0.1
        # 侧脸（yaw 大于20度）时，稍微提高门限
        if yaw is not None and abs(yaw) > 20:
            threshold += 0.1
        return mar > threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 调用 infer() 得到疲劳检测处理后的图像
        processed_frame = infer(frame)
        # 先调用烟雾检测（吸烟检测）
        processed_frame = smoke_infer(processed_frame)

        # 获取头部姿态信息（基于原始 frame）
        pose = hp.estimate_head_pose_frame(frame)
        if pose is not None:
            pitch, yaw, roll = pose
            cv2.putText(processed_frame, f"Pitch: {pitch:.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Yaw: {yaw:.1f}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Roll: {roll:.1f}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            pitch, yaw, roll = None, None, None
            cv2.putText(processed_frame, "No face detected", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 使用新的疲劳判断逻辑：仅当 check_fatigue 返回 True 时，才认为存在疲劳迹象
        if check_fatigue(frame, pitch, yaw):
            fatigue_count += 1
        else:
            fatigue_count = max(0, fatigue_count - 1)

        if fatigue_count >= fatigue_threshold:
            cv2.putText(processed_frame, "Fatigue Detected!", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 绘制“Back”按钮（红色矩形和白色文字）
        cv2.rectangle(processed_frame, button_top_left, button_bottom_right, (0, 0, 255), -1)
        cv2.putText(processed_frame, "Back", (button_top_left[0] + 5, button_top_left[1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Combined Detection", processed_frame)
        videoWriter.write(processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or fatigue_count < 0:
            break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()
    return s_path