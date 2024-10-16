from ultralytics import YOLO
from PIL import Image
import numpy as np
import supervision as sv
from typing import List
from dataclasses import dataclass, field
import cv2

"""
yolov8n-pose.pt 
yolov8s-pose.pt 
yolov8m-pose.pt 
yolov8l-pose.pt 
yolov8x-pose.pt 
yolov8x-pose-p6.pt

tensorboard --logdir runs/pose/train6

이미지를 한장씩 

"""


TRAIN_MODE = True
if TRAIN_MODE:
    model = YOLO('yolov8x-pose.pt')
    results = model.train(data='./test_dataset2/data.yaml', epochs=125000, patience=125000000, imgsz=640, device='cuda:0', batch=4, mosaic=0.0, plots=True,)
else:
    """
앞의 13개는 "#FF1493"로 **분홍색(D eep Pink)**이고, 
뒤의 4개는 "#00BFFF"로 **파란색(Deep Sky Blue)**이야.
    """
    colors: List[str] = [
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", "#00BFFF", "#00BFFF", "#00BFFF"
    ]

    VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
        color=[sv.Color.from_hex(color) for color in colors],
        text_color=sv.Color.from_hex('#FFFFFF'), # white
        border_radius=5,
        text_thickness=1,
        text_scale=0.5,
        text_padding=5,
    )
    labels: List[str] = [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "14", "15", "16"
    ]

    # 모델 로드 (학습한 가중치가 저장된 모델 경로를 사용)
    model = YOLO('runs/pose/train7/weights/best.pt')  # 학습이 끝난 후의 모델 가중치 파일 경로

    # 테스트할 데이터셋의 경로 (학습 시 사용한 test 데이터 경로)
    test_data = './test_dataset2/train/images'  # test 이미지가 있는 경로로 수정

    # 추론 실행
    results = model.predict(source=test_data, save=True, save_txt=True,
                            imgsz=640, conf=0.1,
                            )
    for a_result in results:
        keypoints = sv.KeyPoints.from_ultralytics(a_result)
        print("keypoints.xy.shape:", keypoints.xy.shape) # (3, 16, 2)
        annotated_frame = a_result.orig_img
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, labels)

        cv2.imshow("frame", annotated_frame)

        # Enter 키를 누르면 다음 이미지로 넘어감 (13은 Enter의 ASCII 코드)
        while True:
            if cv2.waitKey(1) == 13:  # Enter 키를 누를 때까지 대기
                break

    # 모든 창 닫기
    cv2.destroyAllWindows()
