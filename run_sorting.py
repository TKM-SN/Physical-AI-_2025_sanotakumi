#!/usr/bin/env python
# coding: utf-8

import cv2 as cv
from speech_color_sorting import speech_color_sorting

# インスタンス生成
sorting = speech_color_sorting()

def main():
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        # YOLO 版処理
        vis = sorting.Sorting_grap_yolo(frame)

        cv.imshow("Color Sorting YOLO", vis)
        if cv.waitKey(1) & 0xFF == 27:  # ESCで終了
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
