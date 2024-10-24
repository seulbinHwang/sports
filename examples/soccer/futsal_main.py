import argparse
from enum import Enum
from typing import Iterator, List

import os
import cv2
import supervision as sv
from ultralytics import YOLO

from sports.annotators.futsal import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team_mobile_clip import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.futsal import FutsalPitchConfiguration
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Union, Any, Optional
from collections import defaultdict

DETECTION_ELAPSED_TIME = 0
tracking_elapsed_time = 0
team_classifier_elapsed_time = 0
render_elapsed_time = 0
# 현재 실행 중인 파이썬 파일이 속한 디렉토리의 절대 경로를 얻는 방법
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR,
                                           'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR,
                                          'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR,
                                         'data/football-ball-detection.pt')
USE_YOLO = True

GOALKEEPER_CLASS_ID = 1
if USE_YOLO:
    BALL_CLASS_ID = 32
    PLAYER_CLASS_ID = 0
else:
    BALL_CLASS_ID = 0
    PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

# Team ID tracking dictionary
STRIDE = 60
CONFIG = FutsalPitchConfiguration()

COLORS = ['#FF0000', '#00FF00', '#FFFFFF', '#000000']

VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
# EDGE_ANNOTATOR = sv.EdgeAnnotator(
#     color=sv.Color.from_hex('#FF1493'),
#     thickness=2,
#     edges=CONFIG.edges,
# )
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(COLORS),
                                thickness=2)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(COLORS),
                                        thickness=2)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#800080'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def get_half_crops(frame: np.ndarray,
                   detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract upper half crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images (upper half of the bounding boxes).
    """
    crops = []
    for xyxy in detections.xyxy:
        x_min, y_min, x_max, y_max = xyxy

        # Calculate mid-point of the bounding box height (for upper half)
        mid_y = y_min + (y_max - y_min) // 2

        # Update bounding box to only include the upper half
        upper_half_box = np.array([x_min, y_min, x_max, mid_y])

        # Crop the image using the updated bounding box (upper half)
        crop = sv.crop_image(frame, upper_half_box)
        crops.append(crop)

    return crops


def resolve_goalkeepers_team_id(players: sv.Detections,
                                players_team_id: np.array,
                                goalkeepers: sv.Detections) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(
        sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(transformed_xy: np.ndarray,
                 color_lookup: np.ndarray) -> np.ndarray:
    radar = draw_pitch(config=CONFIG)

    radar = draw_points_on_pitch(config=CONFIG,
                                 xy=transformed_xy[color_lookup == 0],
                                 face_color=sv.Color.from_hex(COLORS[0]),
                                 radius=5,
                                 pitch=radar)
    radar = draw_points_on_pitch(config=CONFIG,
                                 xy=transformed_xy[color_lookup == 1],
                                 face_color=sv.Color.from_hex(COLORS[1]),
                                 radius=5,
                                 pitch=radar)
    radar = draw_points_on_pitch(config=CONFIG,
                                 xy=transformed_xy[color_lookup == 2],
                                 face_color=sv.Color.from_hex(COLORS[2]),
                                 radius=5,
                                 pitch=radar)
    # ball
    radar = draw_points_on_pitch(config=CONFIG,
                                 xy=transformed_xy[color_lookup == 3],
                                 face_color=sv.Color.from_hex(COLORS[3]),
                                 radius=5,
                                 pitch=radar)
    return radar


def run_pitch_detection(source_video_path: str,
                        device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        # key_points.xy.shape: (1, 32, 2) -> (1, 17, 2)
        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(source_video_path: str,
                         device: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    if USE_YOLO:
        player_detection_model = YOLO("yolo11x.pt").to(device=device)
    else:
        player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(
            device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame,
                                                       detections)
        yield annotated_frame


def run_ball_detection(source_video_path: str,
                       device: str) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(source_video_path: str,
                        device: str) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame,
                                                     detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame,
                                                           detections,
                                                           labels=labels)
        yield annotated_frame


def run_team_classification(source_video_path: str,
                            device: str) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    # crops = []
    # for frame in tqdm(frame_generator, desc='collecting crops'):
    #     result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
    #     detections = sv.Detections.from_ultralytics(result)
    #     crops += get_half_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])
    # sv.plot_images_grid(crops[:50], grid_size=(10, 5))
    # print("here")

    team_classifier = TeamClassifier(device=device)
    # team_classifier.fit(crops)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crop_start_time = time.time()
        crops = get_half_crops(frame, players)
        crop_elapsed_time = time.time() - crop_start_time
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(players_team_id.tolist() +
                                goalkeepers_team_id.tolist() +
                                [REFEREE_CLASS_ID] * len(referees))
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame,
            detections,
            labels,
            custom_color_lookup=color_lookup)
        yield annotated_frame


def get_17_points_from_image(image: np.ndarray) -> np.ndarray:
    """
    이미지(numpy 배열)를 GUI로 띄운 후, 사용자가 17개의 점을 클릭하여 좌표를 저장하는 함수.
    각 점을 클릭할 때 몇 번째 점을 클릭하는지 GUI에 표시되며, 's' 키를 누르면 (-1., -1.) 좌표로 기록된다.
    클릭한 점은 빨간색 점으로 시각화되며, 17개의 점 좌표를 얻으면 GUI가 종료되고, 좌표 배열을 반환한다.

    Args:
        image (np.ndarray): 입력 이미지(numpy 배열 형식).

    Returns:
        np.ndarray: (1, 17, 2) shape의 numpy 배열로 각 점의 좌표가 담긴다.
    """
    points = np.zeros((17, 2))  # 17개의 점을 저장할 배열
    current_point = [0]  # 현재 몇 번째 점인지 저장 (리스트로 만든 이유는 nonlocal로 참조하기 위함)
    skip_flag = [False]  # 's' 키가 눌렸는지 여부를 추적

    def on_click(event):
        """마우스 클릭 이벤트 핸들러로, 좌표를 기록."""
        if current_point[0] < 17 and not skip_flag[0]:
            points[current_point[0]] = [event.xdata, event.ydata]
            ax.plot(event.xdata, event.ydata, 'ro')  # 빨간색 점을 찍음
            current_point[0] += 1
            update_title()
            fig.canvas.draw()  # 화면 업데이트

            if current_point[0] == 17:
                plt.close()  # 17개의 점이 모두 찍히면 GUI 닫기
        else:
            skip_flag[0] = False  # 다음 클릭에서 정상적인 처리를 위해 플래그 초기화

    def on_key_press(event):
        """키보드 입력 이벤트 핸들러로, 'Enter' 키를 눌렀을 때 Skip 동작을 처리."""
        if event.key == 'enter' and current_point[0] < 17:
            points[current_point[0]] = [-1., -1.]
            current_point[0] += 1

            update_title()
            fig.canvas.draw()  # 화면 업데이트

            if current_point[0] == 17:
                plt.close()  # 17개의 점이 모두 찍히면 GUI 닫기

    def update_title():
        """현재 몇 번째 점을 찍고 있는지 타이틀 업데이트."""
        title.set_text(
            f'Time to click point {current_point[0]+1}/17 (Press \'s\' to skip)'
        )

    # GUI 창 띄우기
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)  # 버튼 공간 확보를 위해 조정
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb_image)
    title = ax.set_title('Time to click point 1/17 (Press \'s\' to skip)')

    # 클릭 이벤트 연결
    fig.canvas.mpl_connect('button_press_event', on_click)

    # 키보드 이벤트 연결 ('s' 키로 skip 처리)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    plt.show()

    # (1, 17, 2) shape으로 좌표 반환
    return np.expand_dims(points, axis=0)


def get_manual_keypoints(image: np.ndarray) -> sv.KeyPoints:
    # xy = get_17_points_from_image(image)
    xy = np.array([[[-1, -1], [13.048, 608.02], [86.597, 666.08],
                    [162.08, 728.02], [640.15, 1078.3], [-1, -1], [-1, -1],
                    [1102.7, 424.15], [-1, -1], [1387.2, 418.34],
                    [1849.8, 424.15], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
                    [-1, -1], [-1, -1]]])
    class_id = np.array([0])
    class_names = np.array(['pitch'])
    data = {"class_name": class_names}

    return sv.KeyPoints(xy, class_id, data=data)


def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    if USE_YOLO:
        player_detection_model = YOLO("yolo11x.pt").to(device=device)
    else:
        player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(
            device=device)
    # pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    team_classifier = TeamClassifier(device=device)

    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    keypoints = None

    for frame in frame_generator:
        # result = pitch_detection_model(frame, verbose=False)[0]
        # keypoints = sv.KeyPoints.from_ultralytics(result)
        if keypoints is None:
            keypoints = get_manual_keypoints(frame)
        result = player_detection_model(frame,
                                        classes = [PLAYER_CLASS_ID, BALL_CLASS_ID],
                                        imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        ########################
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
        transformer = ViewTransformer(
            source=keypoints.xy[0][mask].astype(np.float32),
            target=np.array(CONFIG.vertices)[mask].astype(np.float32))
        xy = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER)
        transformed_xy = transformer.transform_points(
            points=xy)  # shape: (n_players, 2)
        """
        valid transformed_xy:  
            if player, 0.0 <= x <= w, 0.0 <= y <= h is valid
            else, all valid
        """
        h, w, _ = frame.shape
        non_player_valid_mask = detections.class_id != PLAYER_CLASS_ID  # (n_players,)
        x_limit = (CONFIG.length / 2)
        y_limit = (CONFIG.width / 2)
        player_valid_mask = (
            (np.abs(transformed_xy[:, 0])<= x_limit) & (np.abs(transformed_xy[:, 1]) <= y_limit) &
            (detections.class_id == PLAYER_CLASS_ID))  # (n_players,)
        valid_mask = non_player_valid_mask | player_valid_mask  # (n_players,)
        # remove invalid players from detections.
        print("--------------------")
        print("[before]detections:", detections.class_id, "len:", len(detections.class_id))
        detections = detections[valid_mask]
        print("[after]detections:", detections.class_id, "len:", len(detections.class_id))
        detections = tracker.update_with_detections(detections)
        print("[track]detections:", detections.class_id, "len:", len(detections.class_id))
        ########################
        ball = detections[detections.class_id == BALL_CLASS_ID]
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        player_crops = get_half_crops(frame, players)
        players_team_id = team_classifier.predict(
            player_crops)  # shape: (n_players,)
        detections = sv.Detections.merge([players, ball])
        xy = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER)
        transformed_xy = transformer.transform_points(
            points=xy)  # shape: (n_players, 2)
        color_lookup = np.array(players_team_id.tolist() + [3] * len(ball))
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame,
            detections,
            labels,
            custom_color_lookup=color_lookup)
        h, w, _ = frame.shape

        radar = render_radar(transformed_xy, color_lookup)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(x=w // 2 - radar_w // 2,
                       y=h - radar_h,
                       width=radar_w,
                       height=radar_h)
        annotated_frame = sv.draw_image(annotated_frame,
                                        radar,
                                        opacity=0.5,
                                        rect=rect)
        yield annotated_frame


def main(source_video_path: str, target_video_path: str, device: str,
         mode: Mode) -> None:
    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.RADAR:
        frame_generator = run_radar(source_video_path=source_video_path,
                                    device=device)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    frame_number = 0
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            frame_number += 1
            print(f"Frame {frame_number}")
            sink.write_frame(frame)

            # cv2.imshow("frame", frame)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break
        cv2.destroyAllWindows()
    print(f"Processed {frame_number} frames.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=Mode, default=Mode.RADAR)
    args = parser.parse_args()
    srtart_time = time.time()
    main(source_video_path=args.source_video_path,
         target_video_path=args.target_video_path,
         device=args.device,
         mode=args.mode)
    print(f"Elapsed time: {time.time() - srtart_time:.2f} seconds")
"""
python main.py --source_video_path data/2e57b9_0.mp4 \
--target_video_path data/2e57b9_0-radar.mp4 \
--device cuda --mode RADAR


python -m examples.soccer.main --source_video_path examples/soccer/data/short_2e57b9_0.mp4 \
--target_video_path examples/soccer/data/short_2e57b9_0-radar.mp4 \
--device cuda --mode RADAR

python -m examples.soccer.futsal_main --source_video_path examples/soccer/data/short_input.mp4 \
--target_video_path examples/soccer/data/short_input-radar.mp4 \
--device cuda:0 --mode RADAR

python -m examples.soccer.futsal_main --source_video_path examples/soccer/data/short_input.mp4 \
--target_video_path examples/soccer/data/short_input-radar.mp4 \
--device mps --mode RADAR

python -m examples.soccer.futsal_main --source_video_path examples/soccer/data/input.mp4 \
--target_video_path examples/soccer/data/input-pitch-detection.mp4 \
--device cuda --mode PITCH_DETECTION

python -m examples.soccer.futsal_main --source_video_path examples/soccer/data/short_input.mp4 \
--target_video_path examples/soccer/data/short_input-pitch-detection.mp4 \
--device cuda:0 --mode PITCH_DETECTION


python -m examples.soccer.futsal_main --source_video_path examples/soccer/data/short_input.mp4 \
--target_video_path examples/soccer/data/short_input-ball.mp4 \
--device mps --mode BALL_DETECTION


python -m examples.soccer.futsal_main --source_video_path examples/soccer/data/short_input.mp4 \
--target_video_path examples/soccer/data/short_input-player-detection.mp4 \
--device cuda:0 --mode PLAYER_DETECTION

python -m examples.soccer.futsal_main --source_video_path examples/soccer/data/short_input.mp4 \
--target_video_path examples/soccer/data/short_input-player-detection.mp4 \
--device mps --mode PLAYER_DETECTION

python -m examples.soccer.futsal_main --source_video_path examples/soccer/data/short_input.mp4 \
--target_video_path examples/soccer/data/short_input-team-classification.mp4 \
--device cuda:0 --mode TEAM_CLASSIFICATION

ffmpeg -ss 00:00:22 -to 00:00:30 -i vlog.mp4 -c copy short_vlog.mp4
ffmpeg -ss 00:02:00 -to 00:02:05 -i long_output.mp4 -c copy short_output.mp4


ffmpeg -ss 00:00:00 -to 00:00:01 -i input.mp4 -c copy short_input.mp4


ffmpeg -i 1.MOV -vf "fps=3" new/1_%04d.jpg
ffmpeg -i 2.MOV -vf "fps=3" new/2_%04d.jpg
ffmpeg -i 3.MOV -vf "fps=3" new/3_%04d.jpg
ffmpeg -i 4.MOV -vf "fps=3" new/4_%04d.jpg
ffmpeg -i 5.MOV -vf "fps=3" new/5_%04d.jpg


--------------------------
32 frame 

총 소요 시간: 46.86 초 (1장당 1.46 초)

detection: 3.69초 (1장당 0.115초)
팀 구분 : 6.73초  (1장당 0.23초)

6.42 (siglip)
5.74 (mobile clip)


------------------
30초 영상 (900장)

팀 구분:  94.43 초 -> 9.53 fps


108.41 (siglip2)
107.92 (siglip)


"""
