import cv2
import dlib
import numpy as np

# 初始化人脸检测器和关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 3D模型人脸关键点（通用模型）
object_pts = np.array([
    [0.0, 0.0, 0.0],             # 鼻尖
    [0.0, -330.0, -65.0],        # 下巴
    [-225.0, 170.0, -135.0],     # 左眼左角
    [225.0, 170.0, -135.0],      # 右眼右角
    [-150.0, -150.0, -125.0],    # 左嘴角
    [150.0, -150.0, -125.0]      # 右嘴角
])

# 获取头部姿态估计
def get_head_pose(shape, cam_matrix, dist_coeffs=np.zeros((4, 1))):
    image_pts = np.array([
        (shape.part(30).x, shape.part(30).y),  # 鼻尖
        (shape.part(8).x, shape.part(8).y),    # 下巴
        (shape.part(36).x, shape.part(36).y),  # 左眼角
        (shape.part(45).x, shape.part(45).y),  # 右眼角
        (shape.part(48).x, shape.part(48).y),  # 左嘴角
        (shape.part(54).x, shape.part(54).y)   # 右嘴角
    ], dtype='double')

    # 姿态估计，solvePnP解算
    success, rotation_vec, translation_vec = cv2.solvePnP(
        object_pts, image_pts, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    # 转换旋转向量为旋转矩阵
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)

    # 旋转矩阵转欧拉角
    proj_mat = np.hstack((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_mat)

    pitch, yaw, roll = [angle[0] for angle in euler_angles]
    
    return pitch, yaw, roll

def estimate_head_pose_frame(frame):
    # 获取图像尺寸及相机参数
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    cam_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    
    # 转换为灰度图像并检测人脸
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    
    if len(faces) > 0:
        face = faces[0]
        shape = predictor(gray, face)
        return get_head_pose(shape, cam_matrix)
    else:
        return None

# 主函数（测试用）
def estimate_head_pose(video_source=0):
    cap = cv2.VideoCapture(video_source)

    # 摄像头参数（示例，最好自行校准）
    focal_length = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    center = (focal_length / 2, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
    cam_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces:
            shape = predictor(gray, face)
            pitch, yaw, roll = get_head_pose(shape, cam_matrix)

            # 显示角度信息
            cv2.putText(frame, f"Pitch: {int(pitch)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Yaw: {int(yaw)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Roll: {int(roll)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # 绘制脸部关键点
            for i in range(68):
                pt = (shape.part(i).x, shape.part(i).y)
                cv2.circle(frame, pt, 1, (0, 0, 255), -1)

        cv2.imshow("Head Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # 按ESC退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    estimate_head_pose()

def calculate_mar(frame):
    """
    计算嘴部宽高比（Mouth Aspect Ratio, MAR），用于衡量张嘴程度。
    利用 dlib 检测人脸后，使用内嘴唇关键点：
    公式示例：MAR = (||p62 - p66|| + ||p63 - p65||) / (2 * ||p60 - p64||)
    注意：dlib 的 landmark 索引从0开始，对应 68 点模型时，
    p60: shape.part(60), p64: shape.part(64),
    p62: shape.part(62), p66: shape.part(66),
    p63: shape.part(63), p65: shape.part(65)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    if len(faces) > 0:
        face = faces[0]
        shape = predictor(gray, face)
        # 计算垂直距离
        A = np.linalg.norm(
            np.array([shape.part(62).x, shape.part(62).y]) -
            np.array([shape.part(66).x, shape.part(66).y])
        )
        B = np.linalg.norm(
            np.array([shape.part(63).x, shape.part(63).y]) -
            np.array([shape.part(65).x, shape.part(65).y])
        )
        # 计算水平距离
        C = np.linalg.norm(
            np.array([shape.part(60).x, shape.part(60).y]) -
            np.array([shape.part(64).x, shape.part(64).y])
        )
        mar = (A + B) / (2.0 * C)
        return mar
    else:
        return None