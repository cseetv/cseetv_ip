\# cseetv



야간 CCTV 영상에서 움직임을 감지하는 영상처리 프로젝트입니다.



\## 주요 기능



\- 영상 밝기 자동 보정

\- 노이즈 제거

\- 프레임 차이 기반 움직임 감지

\- 움직임 감지 시 감지 영역 표시

\- 감지 시간 로그 출력

\- 초당 지정된 횟수만 프레임 검사



\## 실행 방법



필요한 라이브러리를 설치합니다.



```bash

pip install opencv-python numpy

```



영상 파일 이름을 `test\_video.mp4`로 설정한 뒤 실행합니다.



```bash

python night\_motion\_detector.py

```



윈도우에서는 아래 명령어도 사용할 수 있습니다.



```bash

py night\_motion\_detector.py

```



\## 현재 구현 상태



현재는 영상처리 코드만 구현되어 있습니다.



추후 추가 예정 기능:



\- Streamlit 기반 웹 화면

\- 실시간 카메라 연결

\- 움직임 감지 시 사용자 알림

