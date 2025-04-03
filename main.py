import sys
import subprocess
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtMultimedia import QSoundEffect
import head_pose as hp
import onnx_infer as oi  # 要求 infer(frame) -> (fatigue_frame, found_fatigue)
                        # 以及 smoke_infer(frame) -> smoke_frame

def is_dark_mode():
    try:
        result = subprocess.check_output(["defaults", "read", "-g", "AppleInterfaceStyle"]).strip()
        return result == b"Dark"
    except Exception:
        return False

# 信号对象，用于检测任务
class WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(np.ndarray, bool, bool)

# 信号对象，用于指标任务
class MetricsSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(str, float, float)

# 检测任务：疲劳与烟雾检测
class DetectionTask(QtCore.QRunnable):
    def __init__(self, frame):
        super().__init__()
        self.frame = frame
        self.signals = WorkerSignals()
    
    def run(self):
        try:
            fatigue_frame, found_fatigue = oi.infer(self.frame)
            smoke_frame = oi.smoke_infer(self.frame)
            lower_red = np.array([0, 0, 200])
            upper_red = np.array([50, 50, 255])
            mask = cv2.inRange(smoke_frame, lower_red, upper_red)
            found_smoke = cv2.countNonZero(mask) > 50

            combined_frame = np.maximum(fatigue_frame, smoke_frame) if found_smoke else fatigue_frame
            self.signals.finished.emit(combined_frame, found_fatigue, found_smoke)
        except Exception as e:
            print("DetectionTask error:", e)

# 指标任务：计算 head pose、EAR 与 MAR
class MetricsTask(QtCore.QRunnable):
    def __init__(self, frame):
        super().__init__()
        self.frame = frame
        self.signals = MetricsSignals()
    
    def run(self):
        head_pose = hp.estimate_head_pose_frame(self.frame)
        if head_pose is not None:
            pitch, yaw, roll = head_pose
            head_pose_str = f"Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}"
        else:
            head_pose_str = "No face detected"
        # 这里依然调用 EAR 和 MAR 计算，如需禁用，可直接返回固定值
        ear = hp.calculate_ear(self.frame)
        mar = hp.calculate_mar(self.frame)
        self.signals.finished.emit(head_pose_str, ear if ear is not None else 0.0, mar if mar is not None else 0.0)

# 摄像头显示控件
class RoundedVideoWidget(QtWidgets.QWidget):
    headPoseSignal = QtCore.pyqtSignal(str)
    fatigueAlarmSignal = QtCore.pyqtSignal(bool)
    smokeAlarmSignal = QtCore.pyqtSignal(bool)
    metricsSignal = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # 降低采集帧率到 50ms（约 20 FPS）
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.capture_frame)
        self.timer.start(50)
        self._pixmap = None
        self.detection_mode = False
        self.detection_running = False
        # 用于节流指标计算的帧计数器
        self.frame_count = 0
        # 保存上一次更新的指标数据，减少不必要的UI刷新
        self._last_head_pose = None
        self._last_ear = None
        self._last_mar = None

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.frame_count += 1

        # 每 3 帧异步计算指标
        if self.frame_count % 4 == 0:
            metrics_task = MetricsTask(frame)
            metrics_task.signals.finished.connect(self.handle_metrics)
            QtCore.QThreadPool.globalInstance().start(metrics_task)

        # 如果启用检测模式且没有任务正在运行，则启动检测任务
        if self.detection_mode and not self.detection_running:
            self.detection_running = True
            task = DetectionTask(frame)
            task.signals.finished.connect(self.handle_detection_result)
            QtCore.QThreadPool.globalInstance().start(task)

        # 未启用检测模式时，直接更新显示原始帧
        if not self.detection_mode:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h_img, w_img, ch = rgb_image.shape
            bytes_per_line = ch * w_img
            qt_image = QtGui.QImage(rgb_image.data, w_img, h_img, bytes_per_line, QtGui.QImage.Format_RGB888)
            self._pixmap = QtGui.QPixmap.fromImage(qt_image)
            self.update()

    def handle_detection_result(self, processed_frame, found_fatigue, found_smoke):
        self.detection_running = False
        self.fatigueAlarmSignal.emit(found_fatigue)
        self.smokeAlarmSignal.emit(found_smoke)
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h_img, w_img, ch = rgb_image.shape
        bytes_per_line = ch * w_img
        qt_image = QtGui.QImage(rgb_image.data, w_img, h_img, bytes_per_line, QtGui.QImage.Format_RGB888)
        if self._pixmap is None or self._pixmap.toImage().bits() != qt_image.bits():
            self._pixmap = QtGui.QPixmap.fromImage(qt_image)
            self.update()

    def handle_metrics(self, head_pose_str, ear, mar):
        if (self._last_head_pose != head_pose_str or 
            self._last_ear != ear or 
            self._last_mar != mar):
            self._last_head_pose = head_pose_str
            self._last_ear = ear
            self._last_mar = mar
            self.headPoseSignal.emit(head_pose_str)
            self.metricsSignal.emit(ear, mar)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        radius = 30
        rectF = QtCore.QRectF(self.rect())
        path = QtGui.QPainterPath()
        path.addRoundedRect(rectF, radius, radius)
        painter.setClipPath(path)
        if self._pixmap:
            scaled_pixmap = self._pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            x = int((self.width() - scaled_pixmap.width()) / 2)
            y = int((self.height() - scaled_pixmap.height()) / 2)
            painter.drawPixmap(x, y, scaled_pixmap)

    def close(self):
        self.cap.release()
        super().close()

# 报警覆盖层（红光闪烁背景 + 文本，文本向下偏移20px）
class AlarmOverlay(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.fatigue_text = ""
        self.smoke_text = ""
        self.hide()
        self.bg = FlashingBackground(self)
        self.bg.setGeometry(self.rect())
        self.hide_delay_timer = QtCore.QTimer(self)
        self.hide_delay_timer.setSingleShot(True)
        self.hide_delay_timer.timeout.connect(self._stop_alarm)

    def set_alarm_text(self, fatigue_text, smoke_text):
        if fatigue_text or smoke_text:
            # New alarm triggered: cancel any pending hide timer and update texts immediately.
            if self.hide_delay_timer.isActive():
                self.hide_delay_timer.stop()
            self.fatigue_text = fatigue_text
            self.smoke_text = smoke_text
            self.show()
            self.bg.start_blinking()
        else:
            # No new alarm: do not update texts immediately.
            # Keep the current texts visible until the delay timer expires.
            if not self.hide_delay_timer.isActive():
                self.hide_delay_timer.start(1000)  # Delay of 1 second before clearing
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.bg.setGeometry(self.rect())

    def paintEvent(self, event):
        if not (self.fatigue_text or self.smoke_text):
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        font = QtGui.QFont("Arial", 28, QtGui.QFont.Bold)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtCore.Qt.white))
        texts = []
        if self.fatigue_text:
            texts.append(self.fatigue_text)
        if self.smoke_text:
            texts.append(self.smoke_text)
        combined_text = " | ".join(texts)
        painter.drawText(self.rect().adjusted(0, 20, 0, 0), QtCore.Qt.AlignCenter, combined_text)

    def _stop_alarm(self):
        self.bg.stop_blinking()
        # Clear the alarm texts after the delay
        self.fatigue_text = ""
        self.smoke_text = ""
        self.hide()
class FlashingBackground(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.effect = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.effect)
        self.animation = QtCore.QPropertyAnimation(self.effect, b"opacity")
        self.animation.setDuration(500)
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.setLoopCount(-1)

    def start_blinking(self):
        if self.animation.state() != QtCore.QAbstractAnimation.Running:
            self.animation.start()

    def stop_blinking(self):
        self.animation.stop()
        self.effect.setOpacity(1.0)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        gradient = QtGui.QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QtGui.QColor(255, 0, 0, 255))
        gradient.setColorAt(1, QtGui.QColor(255, 0, 0, 0))
        painter.fillRect(self.rect(), gradient)

# 主窗口
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_dark_mode = is_dark_mode()
        self.setWindowTitle("Fatigue and Smoke Detection")
        self.resize(800, 700)
        palette = QtWidgets.QApplication.palette()
        self.setStyleSheet(f"background-color: {palette.window().color().name()};")

        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        main_vlayout = QtWidgets.QVBoxLayout(central_widget)
        main_vlayout.setContentsMargins(20, 20, 20, 20)
        main_vlayout.setSpacing(20)
        

        # 菜单栏
        menubar = self.menuBar()
        upload_menu = menubar.addMenu("UPLOAD")

        upload_image_action = QtWidgets.QAction("Upload pictures", self)
        upload_video_action = QtWidgets.QAction("Upload video", self)

# 将动作添加到菜单
        upload_menu.addAction(upload_image_action)
        upload_menu.addAction(upload_video_action)

# 连接信号到已有的上传处理函数
        upload_image_action.triggered.connect(self.upload_image)
        upload_video_action.triggered.connect(self.upload_video)
        # # 新增：文件上传功能
        # self.upload_bar_layout = QtWidgets.QHBoxLayout()
        # self.upload_bar_layout.setAlignment(QtCore.Qt.AlignLeft)
        # main_vlayout.insertLayout(0, self.upload_bar_layout)
        
        # self.fold_button = QtWidgets.QToolButton()
        # self.fold_button.setText("☰")
        # self.fold_button.setCheckable(True)
        # self.fold_button.setFixedSize(30, 30)
        # self.fold_button.clicked.connect(self.toggle_upload_panel)
        # self.upload_bar_layout.addWidget(self.fold_button)
        
        # self.upload_panel = QtWidgets.QWidget()
        # upload_panel_layout = QtWidgets.QVBoxLayout(self.upload_panel)  # 这里改为垂直布局
        # self.upload_panel.setVisible(False)

        # self.upload_image_btn = QtWidgets.QPushButton("上传图片")
        # self.upload_video_btn = QtWidgets.QPushButton("上传视频")

        # upload_panel_layout.addWidget(self.upload_image_btn)
        # upload_panel_layout.addWidget(self.upload_video_btn)

        # self.upload_bar_layout.addWidget(self.upload_panel)
        
        # self.upload_image_btn.clicked.connect(self.upload_image)
        # self.upload_video_btn.clicked.connect(self.upload_video)

        # 将 head pose 信息与指标放在同一行，紧挨着居中显示
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setSpacing(5)
        top_layout.setAlignment(QtCore.Qt.AlignCenter)
        self.info_label = QtWidgets.QLabel()
        self.info_label_base_style = "color: white; font-size: 20px; background-color: none;"
        self.info_label.setStyleSheet(self.info_label_base_style)
        self.info_label.setAlignment(QtCore.Qt.AlignCenter)
        top_layout.addWidget(self.info_label)
        self.metrics_label = QtWidgets.QLabel()
        self.metrics_label.setStyleSheet("color: white; font-size: 18px; background-color: none;")
        self.metrics_label.setAlignment(QtCore.Qt.AlignCenter)
        top_layout.addWidget(self.metrics_label)
        main_vlayout.addLayout(top_layout)

        self.video_widget = RoundedVideoWidget()
        self.video_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        main_vlayout.addWidget(self.video_widget, stretch=1)

        self.alarm_overlay = AlarmOverlay(self.centralWidget())
        self.alarm_overlay.setGeometry(0, 0, self.centralWidget().width(), 150)
        self.alarm_overlay.raise_()

        self.video_widget.headPoseSignal.connect(self.update_info)
        self.video_widget.fatigueAlarmSignal.connect(self.update_fatigue_alarm)
        self.video_widget.smokeAlarmSignal.connect(self.update_smoke_alarm)
        self.video_widget.metricsSignal.connect(self.update_metrics)

        btn_layout = QtWidgets.QHBoxLayout()
        main_vlayout.addLayout(btn_layout)
        self.fatigue_btn = QtWidgets.QPushButton("Start Detection")
        self.quit_btn = QtWidgets.QPushButton("❌ Quit")
        self.fatigue_btn.setFixedWidth(400)
        self.quit_btn.setFixedWidth(400)
        btn_layout.addWidget(self.fatigue_btn)
        btn_layout.addWidget(self.quit_btn)

        btn_style = """
        QPushButton {
            border-radius: 20px;
            padding: 12px 24px;
            color: white;
            font-size: 15px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #663399, stop:1 #cc66ff);
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #5e3a8f, stop:1 #c084fc);
        }
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:1, x2:1, y2:0, stop:0 #522e80, stop:1 #b368e2);
        }
        """
        self.fatigue_btn.setStyleSheet(btn_style)
        self.quit_btn.setStyleSheet(btn_style)

        self.fatigue_btn.clicked.connect(self.toggle_detection)
        self.quit_btn.clicked.connect(self.close)

        self.alarm_sound = QSoundEffect(self)
        self.alarm_sound.setSource(QtCore.QUrl.fromLocalFile("alarm.wav"))
        self.alarm_sound.setLoopCount(QSoundEffect.Infinite)
        self.alarm_sound.setVolume(0.5)

        self.fatigue_active = False
        self.smoke_active = False
        self.fatigue_count = 0

    def changeEvent(self, event):
        if event.type() in (QtCore.QEvent.ApplicationPaletteChange, QtCore.QEvent.StyleChange):
            palette = QtWidgets.QApplication.palette()
            self.setStyleSheet(f"background-color: {palette.window().color().name()};")
        super().changeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Adjust the alarm overlay to always cover the video widget
        self.alarm_overlay.setGeometry(0, 0, self.centralWidget().width(), 150)
        self.alarm_overlay.raise_()

    def update_info(self, text):
        self.info_label.setText(text)

    def update_metrics(self, ear, mar):
        self.metrics_label.setText(f"EAR: {ear:.2f}   MAR: {mar:.2f}")

    def update_fatigue_alarm(self, flag):
        if flag:
            self.fatigue_count += 1
        else:
            # Optionally, decrement or reset the counter when not detected
            self.fatigue_count = max(0, self.fatigue_count - 1)
        # Only consider fatigue active if closed-eye is detected for 3 consecutive frames
        if self.fatigue_count >= 3:
            self.fatigue_active = True
        else:
            self.fatigue_active = False
        self.update_alarm_overlay()

    def update_smoke_alarm(self, flag):
        self.smoke_active = flag
        self.update_alarm_overlay()

    def update_alarm_overlay(self):
        fatigue_text = "FATIGUE DETECTED" if self.fatigue_active else ""
        smoke_text = "SMOKE DETECTED" if self.smoke_active else ""
        self.alarm_overlay.set_alarm_text(fatigue_text, smoke_text)
        if fatigue_text or smoke_text:
            if not self.alarm_sound.isPlaying():
                self.alarm_sound.play()
        else:
            self.alarm_sound.stop()

    def toggle_upload_panel(self):
        if self.fold_button.isChecked():
            self.upload_panel.setVisible(True)
        else:
            self.upload_panel.setVisible(False)

    def upload_image(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "上传图片", "", "Image Files (*.png *.jpg *.jpeg)", options=options)
        if fileName:
            import cv2
            image = cv2.imread(fileName)
            processed_image, found_fatigue = oi.infer(image)
            self.show_processed_image(processed_image)

    def show_processed_image(self, image):
        import cv2
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, ch = rgb_image.shape
        bytes_per_line = ch * width
        qimg = QtGui.QImage(rgb_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("检测结果 - 图片")
        vbox = QtWidgets.QVBoxLayout(dlg)
        label = QtWidgets.QLabel()
        label.setPixmap(pixmap.scaled(640, 480, QtCore.Qt.KeepAspectRatio))
        vbox.addWidget(label)
        dlg.exec_()

    def upload_video(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "上传视频", "", "Video Files (*.mp4 *.avi *.mov)", options=options)
        if fileName:
            import emotion_recognize as er
            output_path = fileName.replace(".", "_processed.")
            er.process_uploaded_video(fileName, output_path)
            QtWidgets.QMessageBox.information(self, "处理完成", f"处理完成的视频已保存到：{output_path}")

    def toggle_detection(self):
        if self.video_widget.detection_mode:
            self.video_widget.detection_mode = False
            self.fatigue_btn.setText("Start Detection")
            self.fatigue_active = False
            self.smoke_active = False
            self.update_alarm_overlay()
        else:
            self.video_widget.detection_mode = True
            self.fatigue_btn.setText("Stop Detection")

    def closeEvent(self, event):
        self.video_widget.close()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())