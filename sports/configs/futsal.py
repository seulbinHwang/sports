from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class FutsalPitchConfiguration:
    width: int = 2000  # [cm]
    length: int = 4000  # [cm]
    penalty_box_radius: int = 600  # [cm]
    goal_box_width: int = 300  # [cm]
    centre_circle_radius: int = 300  # [cm]

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        return [
            (-self.length/2, -self.width/2),  # 1
            (-self.length/2, -(self.goal_box_width/2 + self.penalty_box_radius)),  # 2
            (-self.length/2, -self.goal_box_width/2),  # 3
            (-self.length/2, self.goal_box_width/2),  # 4
            (-self.length / 2,
             (self.goal_box_width / 2 + self.penalty_box_radius)),  # 5
            (-self.length/2, self.width/2),  # 6
            (0., -self.width/2),  # 7
            (0., -self.centre_circle_radius),  # 8
            (0., 0.),  # 9
            (0., self.centre_circle_radius),  # 10
            (0., self.width/2),  # 11
            (self.length/2, -self.width/2),  # 12
            (self.length/2, -(self.goal_box_width/2 + self.penalty_box_radius)),  # 13
            (self.length/2, -self.goal_box_width/2),  # 14
            (self.length/2, self.goal_box_width/2),  # 15
            (self.length / 2,
             (self.goal_box_width / 2 + self.penalty_box_radius)),  # 16
            (self.length/2, self.width/2),  # 17


        ]

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 6), (7, 11), (12, 17),
        (1, 12), (6, 17)

    ])

    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "15", "16", "17"
    ])

    colors: List[str] = field(default_factory=lambda: [
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF"
    ])
