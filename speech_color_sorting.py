#!/usr/bin/env python
# coding: utf-8
"""
YOLO v8 による色識別付きピック＆プレース
改変元: Yahboom DOFBOT_SE 配布コード
"""

import threading
import time
from time import sleep

import cv2 as cv
import Arm_Lib
from ultralytics import YOLO

# ───────── YOLO 設定 ─────────
YOLO_WEIGHTS = '/home/yahboom/dofbot_ws/yolo_weights/color_blocks.pt'  # 重みファイル
YOLO_CLASSES = {'red', 'green', 'blue', 'yellow'}
YOLO_CONF = 0.25
YOLO_IOU = 0.45
# ────────────────────────────


class speech_color_sorting:
    def __init__(self):
        # 画像バッファ
        self.image = None
        # 連続検出回数カウンタ
        self.num = 0
        # アーム動作状態
        self.status = 'waiting'

        # Yahboom SDK
        self.arm = Arm_Lib.Arm_Device()

        # 物体把持用角度
        self.grap_joint = 135
        self.joints = [90, 40, 30, 67, 90, 30]

        # YOLO モデル読み込み（GPU があれば自動利用）
        self.yolo = YOLO(YOLO_WEIGHTS)

    # ───────────────────────── YOLO検出部
    def detect_by_yolo(self, frame):
        """
        BGR フレーム → [(label, (cx, cy), conf), ...]
        """
        results = self.yolo(frame, verbose=False,
                            imgsz=640, conf=YOLO_CONF, iou=YOLO_IOU)[0]
        outs = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.yolo.names[cls_id]
            if label not in YOLO_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            outs.append((label, (int(cx), int(cy)), float(box.conf)))
        return outs

    # ───────────────────────── メイン処理
    def Sorting_grap_yolo(self, img):
        """
        YOLO で色ブロックを検出し、10 フレーム連続したら掴み動作へ
        """
        self.image = cv.resize(img, (640, 480))
        detects = self.detect_by_yolo(self.image)

        if self.status == 'waiting' and detects:
            # 確信度最大の検出のみ採用
            detects.sort(key=lambda x: x[2], reverse=True)
            label, (cx, cy), conf = detects[0]

            # 可視化（任意）
            cv.circle(self.image, (cx, cy), 5, (0, 255, 0), -1)
            cv.putText(self.image, f'{label} {conf:.2f}',
                       (cx + 5, cy - 5), cv.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0), 2)

            self.num += 1
            if self.num % 10 == 0:
                self.status = 'Runing'
                threading.Thread(target=self.sorting_run,
                                 args=(label,)).start()
                self.num = 0
        else:
            self.num = 0  # 検出が切れたらリセット

        return self.image

    # ───────────────────────── 既存アーム制御ロジック
    def sorting_move(self, joints_target):
        joints_up = [90, 80, 35, 40, 90, self.grap_joint]

        self.arm.Arm_serial_servo_write6_array(joints_up, 500)
        sleep(1)
        self.arm.Arm_serial_servo_write(6, 30, 500)
        sleep(0.5)
        self.arm.Arm_serial_servo_write6_array(self.joints, 1000)
        sleep(1)
        self.arm.Arm_serial_servo_write(6, self.grap_joint, 500)
        sleep(0.5)
        self.arm.Arm_serial_servo_write6_array(joints_up, 1000)
        sleep(1)
        self.arm.Arm_serial_servo_write(1, joints_target[0], 500)
        sleep(0.5)
        self.arm.Arm_serial_servo_write6_array(joints_target, 1000)
        sleep(1.5)
        self.arm.Arm_serial_servo_write(6, 30, 500)
        sleep(0.5)
        joints_up[0] = joints_target[0]
        self.arm.Arm_serial_servo_write6_array(joints_up, 800)
        sleep(0.5)
        self.arm.Arm_serial_servo_write(1, 90, 500)
        sleep(0.5)
        joints_0 = [90, 135, 0, 0, 90, 30]
        self.arm.Arm_serial_servo_write6_array(joints_0, 500)
        sleep(1)

    def sorting_run(self, name):
        if name == "red":
            self.arm.Arm_ask_speech(33); time.sleep(0.1)
            joints_target = [117, 19, 66, 56, 90, self.grap_joint]
            self.sorting_move(joints_target)
            self.arm.Arm_ask_speech(32); time.sleep(0.1)
        elif name == "blue":
            self.arm.Arm_ask_speech(35); time.sleep(0.1)
            joints_target = [44, 66, 20, 28, 90, self.grap_joint]
            self.sorting_move(joints_target)
            self.arm.Arm_ask_speech(32); time.sleep(0.1)
        elif name == "green":
            self.arm.Arm_ask_speech(34); time.sleep(0.1)
            joints_target = [136, 66, 20, 29, 90, self.grap_joint]
            self.sorting_move(joints_target)
            self.arm.Arm_ask_speech(32); time.sleep(0.1)
        elif name == "yellow":
            self.arm.Arm_ask_speech(31); time.sleep(0.1)
            joints_target = [65, 22, 64, 56, 90, self.grap_joint]
            self.sorting_move(joints_target)
            self.arm.Arm_ask_speech(32); time.sleep(0.1)

        self.status = 'waiting'  # 動作完了
