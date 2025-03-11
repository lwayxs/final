import cv2
import os
import head_pose as hp
from onnx_infer import infer, smoke_infer

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
        
      


#调用摄像头函数 #use-camera
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
    cap = cv2.VideoCapture(v_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if s_path is None:
        now_path = os.getcwd()
        s_path = os.path.join(now_path, 'combined_output.avi')

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter(s_path, fourcc, fps, (w, h))

    fatigue_count = 0
    fatigue_threshold = 3 # 正常状态下连续3帧检测到疲劳

    # “Back”按钮区域
    button_top_left = (w - 110, 10)
    button_bottom_right = (w - 10, 40)

    def mouse_callback(event, x, y, flags, param):
        nonlocal fatigue_count
        if event == cv2.EVENT_LBUTTONDOWN:
            if button_top_left[0] <= x <= button_bottom_right[0] and button_top_left[1] <= y <= button_bottom_right[1]:
                fatigue_count = -100  # 触发退出

    cv2.namedWindow("Combined Detection")
    cv2.setMouseCallback("Combined Detection", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 获取头部姿态信息
        head_pose = hp.estimate_head_pose_frame(frame)
        if head_pose is not None:
            pitch, yaw, roll = head_pose
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Roll: {roll:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # 若检测不到，则设默认值（或保留上次检测值，可根据需要调整）
            pitch, yaw, roll = 0, 0, 0
            cv2.putText(frame, "No face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 调用infer函数获取处理后图像与疲劳标志（依赖于1.jpg的闭眼/张嘴检测以及MAR）
        processed_frame, found_fatigue = infer(frame)

        # 根据头部姿态自适应调整疲劳判断阈值
        # 此处逻辑为：
        # 1. 正常状态下，连续3帧检测到疲劳提示
        # 2. 仰头时（pitch > 15），疲劳更明显，要求连续3帧（可调整）
        # 3. 低头时（pitch < -15），由于视角问题，调整MAR阈值在infer中已做，本处降低连续帧要求为2
        # 4. 侧脸（abs(yaw) > 20）时，要求连续2帧
        if head_pose is None:
            dynamic_threshold = fatigue_threshold
        else:
            if pitch > 15:
                dynamic_threshold = 3
            elif pitch < -15:
                dynamic_threshold = 2
            elif abs(yaw) > 20:
                dynamic_threshold = 2
            else:
                dynamic_threshold = fatigue_threshold

        # 更新疲劳计数逻辑
        if found_fatigue:
            fatigue_count += 1
        else:
            fatigue_count = max(0, fatigue_count - 1)

        # 根据动态阈值判断疲劳
        if fatigue_count >= dynamic_threshold:
            cv2.putText(processed_frame, "Fatigue Detected!", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 调用烟雾检测处理（保持原有逻辑）
        processed_frame = smoke_infer(processed_frame)

        # 绘制“Back”按钮
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