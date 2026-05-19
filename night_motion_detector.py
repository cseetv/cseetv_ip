import cv2
import numpy as np
from datetime import datetime


class NightMotionDetector:
    def __init__(
        self,
        min_motion_area=200,
        motion_threshold=15,
        brightness_threshold=80,
        checks_per_second=2,
        save_output=True,
        output_path="output_motion_result.mp4"
    ):
        """
        min_motion_area:
            이 값보다 작은 움직임 영역은 노이즈로 보고 무시합니다.

        motion_threshold:
            프레임 차이를 이진화할 때 사용하는 기준값입니다.

        brightness_threshold:
            현재 코드에서는 호환용으로만 남겨둔 값입니다.
            실제 밝기 보정은 adjust_brightness_adaptive()에서 자동으로 처리합니다.

        checks_per_second:
            1초에 몇 번 움직임 검사를 할지 정합니다.
            예: 2면 1초에 2프레임만 검사합니다.

        save_output:
            결과 영상을 저장할지 여부입니다.

        output_path:
            저장할 결과 영상 파일명입니다.
        """

        self.min_motion_area = min_motion_area
        self.motion_threshold = motion_threshold
        self.brightness_threshold = brightness_threshold
        self.checks_per_second = checks_per_second
        self.save_output = save_output
        self.output_path = output_path

        self.prev_frame = None
        self.video_writer = None

    def adjust_brightness_adaptive(self, frame, target_brightness=120):
        """
        영상의 현재 밝기를 기준으로 자동 밝기 보정을 수행합니다.

        - 너무 어두우면 밝게 보정
        - 너무 밝으면 살짝 어둡게 보정
        - 적당한 밝기면 거의 유지
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_brightness = np.mean(gray)

        brightness_diff = target_brightness - current_brightness

        # 목표 밝기와 차이가 작으면 보정하지 않음
        if abs(brightness_diff) < 15:
            return frame, current_brightness, "normal"

        alpha = 1.0
        beta = brightness_diff * 0.6

        adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # 어두운 영상이면 CLAHE로 대비도 보정
        if current_brightness < 80:
            lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            enhanced_l = clahe.apply(l_channel)
            enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
            adjusted = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            return adjusted, current_brightness, "dark_corrected"

        # 너무 밝은 영상이면 밝기만 낮춤
        if current_brightness > 170:
            return adjusted, current_brightness, "bright_corrected"

        return adjusted, current_brightness, "soft_corrected"

    def reduce_noise(self, frame):
        """
        밝기 보정 후 생길 수 있는 지직거림, 작은 노이즈를 줄입니다.
        """

        denoised = cv2.fastNlMeansDenoisingColored(
            frame,
            None,
            h=7,
            hColor=7,
            templateWindowSize=7,
            searchWindowSize=21
        )

        blurred = cv2.GaussianBlur(
            denoised,
            (5, 5),
            0
        )

        return blurred

    def preprocess_frame(self, frame):
        """
        프레임 전처리 함수입니다.

        1. 현재 밝기 측정
        2. 밝기에 따라 자동 보정
        3. 노이즈 제거
        4. grayscale 변환
        """

        frame, avg_brightness, correction_type = self.adjust_brightness_adaptive(
            frame,
            target_brightness=120
        )

        frame = self.reduce_noise(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame, gray, avg_brightness, correction_type

    def detect_motion(self, current_gray):
        """
        이전 검사 프레임과 현재 검사 프레임의 차이를 이용해서 움직임을 감지합니다.
        """

        motion_detected = False
        motion_boxes = []

        if self.prev_frame is None:
            self.prev_frame = current_gray
            return motion_detected, motion_boxes

        frame_diff = cv2.absdiff(self.prev_frame, current_gray)

        _, threshold_frame = cv2.threshold(
            frame_diff,
            self.motion_threshold,
            255,
            cv2.THRESH_BINARY
        )

        kernel = np.ones((5, 5), np.uint8)

        # 작은 노이즈 제거
        threshold_frame = cv2.morphologyEx(
            threshold_frame,
            cv2.MORPH_OPEN,
            kernel
        )

        # 움직임 영역을 조금 키워서 박스 검출이 잘 되게 함
        threshold_frame = cv2.dilate(
            threshold_frame,
            kernel,
            iterations=2
        )

        contours, _ = cv2.findContours(
            threshold_frame,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_motion_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            motion_boxes.append((x, y, w, h))
            motion_detected = True

        self.prev_frame = current_gray

        return motion_detected, motion_boxes

    def draw_result(self, frame, motion_detected, motion_boxes, avg_brightness, correction_type):
        """
        결과 화면에 움직임 박스, 상태 문구, 밝기 보정 상태를 표시합니다.
        """

        result_frame = frame.copy()

        for x, y, w, h in motion_boxes:
            cv2.rectangle(
                result_frame,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

        if motion_detected:
            status_text = "Motion Detected"
            status_color = (0, 0, 255)
        else:
            status_text = "Normal"
            status_color = (0, 255, 0)

        cv2.putText(
            result_frame,
            status_text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            2
        )

        cv2.putText(
            result_frame,
            f"Brightness: {avg_brightness:.2f}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            result_frame,
            f"Correction: {correction_type}",
            (30, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            result_frame,
            f"Check: {self.checks_per_second} fps",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return result_frame

    def process_video(self, video_source):
        """
        video_source:
            - 영상 파일 경로: "test_video.mp4"
            - 노트북 웹캠: 0
            - 휴대폰 IP 카메라 주소: "http://주소/video"
        """

        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print("영상을 열 수 없습니다.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            fps = 30

        check_interval = int(fps / self.checks_per_second)

        if check_interval < 1:
            check_interval = 1

        print(f"영상 FPS: {fps}")
        print(f"검사 간격: {check_interval}프레임마다 1번 검사")
        print(f"즉, 1초에 약 {self.checks_per_second}번 검사합니다.")
        print("영상 분석을 시작합니다.")
        print("종료하려면 q 키를 누르세요.")

        if self.save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            # 검사한 프레임만 저장하므로 저장 영상 FPS도 checks_per_second로 설정
            self.video_writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.checks_per_second,
                (width, height)
            )

        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # 1초에 checks_per_second번만 검사
            if frame_count % check_interval != 0:
                continue

            print(f"{frame_count}번째 프레임 검사 중...")

            processed_frame, current_gray, avg_brightness, correction_type = self.preprocess_frame(frame)

            motion_detected, motion_boxes = self.detect_motion(current_gray)

            if motion_detected:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[알림] {now} 움직임이 감지되었습니다.")

            result_frame = self.draw_result(
                processed_frame,
                motion_detected,
                motion_boxes,
                avg_brightness,
                correction_type
            )

            cv2.imshow("Night CCTV Motion Detector", result_frame)

            if self.video_writer is not None:
                self.video_writer.write(result_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        cap.release()

        if self.video_writer is not None:
            self.video_writer.release()

        cv2.destroyAllWindows()

        print("영상 분석이 종료되었습니다.")

        if self.save_output:
            print(f"결과 영상 저장 완료: {self.output_path}")


if __name__ == "__main__":
    detector = NightMotionDetector(
        min_motion_area=200,
        motion_threshold=15,
        brightness_threshold=80,
        checks_per_second=2,
        save_output=True,
        output_path="output_motion_result.mp4"
    )

    # 1. 영상 파일로 테스트할 때
    detector.process_video("test_video.mp4")

    # 2. 노트북 웹캠으로 테스트할 때는 위 줄을 주석 처리하고 아래 줄 사용
    # detector.process_video(0)

    # 3. 휴대폰 IP 카메라로 연결할 때는 아래처럼 사용
    # detector.process_video("http://192.168.0.10:8080/video")
