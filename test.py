import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import cv2

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
        title.set_text(f'Time to click point {current_point[0]+1}/17 (Press \'s\' to skip)')

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

# 검은색 빈 이미지 생성
# image_path = create_black_image(640, 480)
black_image = np.zeros((480, 640, 3), dtype=np.uint8)

# 생성된 검은색 이미지에 대해 좌표를 찍는 함수 호출
result = get_17_points_from_image(black_image)
print(result)  # 결과 출력
