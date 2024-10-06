from typing import Optional, List

import cv2
import supervision as sv
import numpy as np

from sports.configs.soccer import SoccerPitchConfiguration


def draw_pitch(
    config: SoccerPitchConfiguration,
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 25,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1
) -> np.ndarray:
    """
    Draws a soccer pitch with specified dimensions, colors, and scale.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        background_color (sv.Color, optional): Color of the pitch background.
            Defaults to sv.Color(34, 139, 34).
        line_color (sv.Color, optional): Color of the pitch lines.
            Defaults to sv.Color.WHITE.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        line_thickness (int, optional): Thickness of the pitch lines in pixels.
            Defaults to 4.
        point_radius (int, optional): Radius of the penalty spot points in pixels.
            Defaults to 8.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.

    Returns:
        np.ndarray: Image of the soccer pitch.
    """
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)

    pitch_image = np.ones(
        (scaled_width + 2 * padding,
         scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)
    x_offset = config.length / 2
    y_offset = config.width / 2
    for start, end in config.edges:
        point1 = (int((config.vertices[start - 1][0] + x_offset) * scale) + padding,
                  int((config.vertices[start - 1][1] + y_offset) * scale) + padding)
        point2 = (int((config.vertices[end - 1][0] + x_offset) * scale) + padding,
                  int((config.vertices[end - 1][1] + y_offset) * scale) + padding)
        cv2.line(
            img=pitch_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )
    ####### draw left goal box
    point1 = (int((config.vertices[3 - 1][0] + x_offset) * scale) + padding,
              int((config.vertices[3 - 1][1] + y_offset) * scale) + padding)
    point2 = (int((config.vertices[3 - 1][0] + x_offset - config.goal_box_width/2) * scale) + padding,
              int((config.vertices[3 - 1][1] + y_offset) * scale) + padding)
    cv2.line(
        img=pitch_image,
        pt1=point1,
        pt2=point2,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )
    point1 = (int((config.vertices[3 - 1][0] + x_offset - config.goal_box_width/2) * scale) + padding,
              int((config.vertices[3 - 1][1] + y_offset) * scale) + padding)
    point2 = (int((config.vertices[4 - 1][0] + x_offset - config.goal_box_width/2) * scale) + padding,
              int((config.vertices[4 - 1][1] + y_offset) * scale) + padding)
    cv2.line(
        img=pitch_image,
        pt1=point1,
        pt2=point2,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )


    point1 = (int((config.vertices[4 - 1][0] + x_offset) * scale) + padding,
              int((config.vertices[4 - 1][1] + y_offset) * scale) + padding)
    point2 = (int((config.vertices[4 - 1][0] + x_offset - config.goal_box_width/2) * scale) + padding,
              int((config.vertices[4 - 1][1] + y_offset) * scale) + padding)
    cv2.line(
        img=pitch_image,
        pt1=point1,
        pt2=point2,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )

    ####### draw right goal box
    point1 = (int((config.vertices[14 - 1][0] + x_offset) * scale) + padding,
              int((config.vertices[14 - 1][1] + y_offset) * scale) + padding)
    point2 = (int((config.vertices[14 - 1][0] + x_offset + config.goal_box_width/2) * scale) + padding,
              int((config.vertices[14 - 1][1] + y_offset) * scale) + padding)
    cv2.line(
        img=pitch_image,
        pt1=point1,
        pt2=point2,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )
    point1 = (int((config.vertices[14 - 1][0] + x_offset + config.goal_box_width/2) * scale) + padding,
              int((config.vertices[14 - 1][1] + y_offset) * scale) + padding)
    point2 = (int((config.vertices[15 - 1][0] + x_offset + config.goal_box_width/2) * scale) + padding,
              int((config.vertices[15 - 1][1] + y_offset) * scale) + padding)
    cv2.line(
        img=pitch_image,
        pt1=point1,
        pt2=point2,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )


    point1 = (int((config.vertices[15 - 1][0] + x_offset) * scale) + padding,
              int((config.vertices[15 - 1][1] + y_offset) * scale) + padding)
    point2 = (int((config.vertices[15 - 1][0] + x_offset + config.goal_box_width/2) * scale) + padding,
              int((config.vertices[15 - 1][1] + y_offset) * scale) + padding)
    cv2.line(
        img=pitch_image,
        pt1=point1,
        pt2=point2,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )


    ######
    centre_circle_center = (
        scaled_length // 2 + padding,
        scaled_width // 2 + padding
    )
    cv2.circle(
        img=pitch_image,
        center=centre_circle_center,
        radius=scaled_circle_radius,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )

    #########################0
    # 호 그리기
    axes = (int(config.penalty_box_radius * scale), int(config.penalty_box_radius * scale))  # 타원의 크기 (반지름)

    # 중심점, 반지름, 각도 설정
    center = (int((config.vertices[14 - 1][0] + x_offset) * scale) + padding,
              int((config.vertices[14 - 1][1] + y_offset) * scale) + padding)
    angle = 0  # 타원의 회전 각도 (타원이 회전된 각도)
    start_angle = 180  # 호의 시작 각도 (시계 방향)
    end_angle = 270  # 호의 끝 각도 (시계 방향)

    # 흰색(255, 255, 255)으로 호를 그림
    cv2.ellipse(pitch_image, center, axes, angle, start_angle, end_angle,
                color=line_color.as_bgr(),
                thickness=line_thickness)
    ########
    point1 = (int((config.vertices[3 - 1][0] + x_offset + config.penalty_box_radius) * scale) + padding,
              int((config.vertices[3 - 1][1] + y_offset) * scale) + padding)
    point2 = (int((config.vertices[4 - 1][0] + x_offset + config.penalty_box_radius) * scale) + padding,
              int((config.vertices[4 - 1][1] + y_offset) * scale) + padding)
    cv2.line(
        img=pitch_image,
        pt1=point1,
        pt2=point2,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )

    #########################1
    # 호 그리기
    axes = (int(config.penalty_box_radius * scale), int(config.penalty_box_radius * scale))  # 타원의 크기 (반지름)

    # 중심점, 반지름, 각도 설정
    center = (int((config.vertices[15 - 1][0] + x_offset) * scale) + padding,
              int((config.vertices[15 - 1][1] + y_offset) * scale) + padding)
    angle = 0  # 타원의 회전 각도 (타원이 회전된 각도)
    start_angle = 90  # 호의 시작 각도 (시계 방향)
    end_angle = 180  # 호의 끝 각도 (시계 방향)

    # 흰색(255, 255, 255)으로 호를 그림
    cv2.ellipse(pitch_image, center, axes, angle, start_angle, end_angle,
                color=line_color.as_bgr(),
                thickness=line_thickness)
    #########################2
    # 호 그리기
    axes = (int(config.penalty_box_radius * scale), int(config.penalty_box_radius * scale))  # 타원의 크기 (반지름)

    # 중심점, 반지름, 각도 설정
    center = (int((config.vertices[3 - 1][0] + x_offset) * scale) + padding,
              int((config.vertices[3 - 1][1] + y_offset) * scale) + padding)
    angle = 0  # 타원의 회전 각도 (타원이 회전된 각도)
    start_angle = 270  # 호의 시작 각도 (시계 방향)
    end_angle = 360  # 호의 끝 각도 (시계 방향)

    # 흰색(255, 255, 255)으로 호를 그림
    cv2.ellipse(pitch_image, center, axes, angle, start_angle, end_angle,
                color=line_color.as_bgr(),
                thickness=line_thickness)
    ########
    point1 = (int((config.vertices[14 - 1][0] + x_offset - config.penalty_box_radius) * scale) + padding,
              int((config.vertices[14 - 1][1] + y_offset) * scale) + padding)
    point2 = (int((config.vertices[15 - 1][0] + x_offset - config.penalty_box_radius) * scale) + padding,
              int((config.vertices[15 - 1][1] + y_offset) * scale) + padding)
    cv2.line(
        img=pitch_image,
        pt1=point1,
        pt2=point2,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )
    #########################3
    # 호 그리기
    axes = (int(config.penalty_box_radius * scale), int(config.penalty_box_radius * scale))  # 타원의 크기 (반지름)

    # 중심점, 반지름, 각도 설정
    center = (int((config.vertices[4 - 1][0] + x_offset) * scale) + padding,
              int((config.vertices[4 - 1][1] + y_offset) * scale) + padding)
    angle = 0  # 타원의 회전 각도 (타원이 회전된 각도)
    start_angle = 0  # 호의 시작 각도 (시계 방향)
    end_angle = 90  # 호의 끝 각도 (시계 방향)

    # 흰색(255, 255, 255)으로 호를 그림
    cv2.ellipse(pitch_image, center, axes, angle, start_angle, end_angle,
                color=line_color.as_bgr(),
                thickness=line_thickness)


    return pitch_image


def draw_points_on_pitch(
    config: SoccerPitchConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 5,
    thickness: int = 2,
    padding: int = 25,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws points on a soccer pitch.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        xy (np.ndarray): Array of points to be drawn, with each point represented by
            its (x, y) coordinates.
        face_color (sv.Color, optional): Color of the point faces.
            Defaults to sv.Color.RED.
        edge_color (sv.Color, optional): Color of the point edges.
            Defaults to sv.Color.BLACK.
        radius (int, optional): Radius of the points in pixels.
            Defaults to 10.
        thickness (int, optional): Thickness of the point edges in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw points on.
            If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with points drawn on it.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )
    x_offset = config.length / 2
    y_offset = config.width / 2
    for point in xy:
        scaled_point = (
            int((point[0] + x_offset) * scale) + padding,
            int((point[1] + y_offset) * scale) + padding
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness
        )

    return pitch


def draw_paths_on_pitch(
    config: SoccerPitchConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws paths on a soccer pitch.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        paths (List[np.ndarray]): List of paths, where each path is an array of (x, y)
            coordinates.
        color (sv.Color, optional): Color of the paths.
            Defaults to sv.Color.WHITE.
        thickness (int, optional): Thickness of the paths in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw paths on.
            If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with paths drawn on it.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    for path in paths:
        scaled_path = [
            (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            )
            for point in path if point.size > 0
        ]

        if len(scaled_path) < 2:
            continue

        for i in range(len(scaled_path) - 1):
            cv2.line(
                img=pitch,
                pt1=scaled_path[i],
                pt2=scaled_path[i + 1],
                color=color.as_bgr(),
                thickness=thickness
            )

        return pitch


def draw_pitch_voronoi_diagram(
    config: SoccerPitchConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.WHITE,
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws a Voronoi diagram on a soccer pitch representing the control areas of two
    teams.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        team_1_xy (np.ndarray): Array of (x, y) coordinates representing the positions
            of players in team 1.
        team_2_xy (np.ndarray): Array of (x, y) coordinates representing the positions
            of players in team 2.
        team_1_color (sv.Color, optional): Color representing the control area of
            team 1. Defaults to sv.Color.RED.
        team_2_color (sv.Color, optional): Color representing the control area of
            team 2. Defaults to sv.Color.WHITE.
        opacity (float, optional): Opacity of the Voronoi diagram overlay.
            Defaults to 0.5.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw the
            Voronoi diagram on. If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with the Voronoi diagram overlay.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    y_coordinates, x_coordinates = np.indices((
        scaled_width + 2 * padding,
        scaled_length + 2 * padding
    ))

    y_coordinates -= padding
    x_coordinates -= padding

    def calculate_distances(xy, x_coordinates, y_coordinates):
        return np.sqrt((xy[:, 0][:, None, None] * scale - x_coordinates) ** 2 +
                       (xy[:, 1][:, None, None] * scale - y_coordinates) ** 2)

    distances_team_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
    distances_team_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)

    min_distances_team_1 = np.min(distances_team_1, axis=0)
    min_distances_team_2 = np.min(distances_team_2, axis=0)

    control_mask = min_distances_team_1 < min_distances_team_2

    voronoi[control_mask] = team_1_color_bgr
    voronoi[~control_mask] = team_2_color_bgr

    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    return overlay
