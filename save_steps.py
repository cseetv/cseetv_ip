"""PPT용 단계별 이미지 저장

tramStation 프레임 1장을 각 처리 단계별로 저장.
사용법: python save_steps.py
"""
import cv2, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.processing.brightness import adjust_brightness
from app.processing.shadow import remove_shadows

INPUT_DIR = "data/tramStation/input"
OUT_DIR = "experiments/results/steps"
FRAME_IDX = 600
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")])
prev_frame = cv2.imread(os.path.join(INPUT_DIR, files[FRAME_IDX - 1]))
curr_frame = cv2.imread(os.path.join(INPUT_DIR, files[FRAME_IDX]))
print(f"프레임: {files[FRAME_IDX]} ({curr_frame.shape})")

step = 0
def save(name, img, desc=""):
    global step
    path = os.path.join(OUT_DIR, f"step{step:02d}_{name}.png")
    cv2.imwrite(path, img)
    print(f"  [{step:02d}] {name}: {desc}")
    step += 1

save("original", curr_frame, "원본")
clahe_frame, ci = adjust_brightness(curr_frame.copy(), target_brightness=120, clahe_clip=2.0, clahe_tile=8)
save("clahe", clahe_frame, f"CLAHE (μ {ci['brightness_before']:.0f}→{ci['brightness_after']:.0f})")
gray_o = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
save("global_he", cv2.cvtColor(cv2.equalizeHist(gray_o), cv2.COLOR_GRAY2BGR), "전역 HE (비교용)")
fnlm = cv2.fastNlMeansDenoisingColored(clahe_frame, None, 7, 7, 7, 21)
save("fastnlmeans", fnlm, "fastNlMeansDenoisingColored")
gaussian = cv2.GaussianBlur(clahe_frame.copy(), (5,5), 0)
save("gaussian", gaussian, "Gaussian Blur 5×5")
median = cv2.medianBlur(gaussian, 5)
save("median", median, "Median Filter 5")

prev_p, _ = adjust_brightness(prev_frame.copy(), target_brightness=120, clahe_clip=2.0, clahe_tile=8)
prev_p = cv2.GaussianBlur(prev_p, (5,5), 0)
prev_gray = cv2.cvtColor(prev_p, cv2.COLOR_BGR2GRAY)
curr_gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
diff = cv2.absdiff(prev_gray, curr_gray)
save("frame_diff", diff, "Frame Difference")
_, binary = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
save("threshold", binary, "이진화 T=20")
kernel = np.ones((5,5), np.uint8)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
save("morph_open", opened, "Morphology Opening")
dilated = cv2.dilate(opened, kernel, iterations=2)
save("morph_dilate", dilated, "Morphology Dilation")
shadow_rm = remove_shadows(dilated.copy(), prev_p, gaussian)
save("shadow_removal", shadow_rm, "Shadow Removal (참고)")
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
res = curr_frame.copy()
for c in contours:
    if cv2.contourArea(c) < 200:
        continue
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(res, (x,y), (x+w,y+h), (0,0,255), 2)
save("detection_result", res, "Frame Diff 감지 결과")

print("\n  Running Average 배경 생성 중...")
bg = None
for i in range(500, FRAME_IDX+1):
    f = cv2.imread(os.path.join(INPUT_DIR, files[i]))
    f,_ = adjust_brightness(f, target_brightness=120, clahe_clip=2.0, clahe_tile=8)
    f = cv2.GaussianBlur(f, (5,5), 0)
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if bg is None: bg = g.copy()
    else: cv2.accumulateWeighted(g, bg, 0.05)
bg8 = bg.astype(np.uint8)
save("running_avg_bg", bg8, "Running Average 배경 모델")
ra_diff = cv2.absdiff(curr_gray, bg8)
save("running_avg_diff", ra_diff, "RunAvg vs 현재")
_, ra_bin = cv2.threshold(ra_diff, 20, 255, cv2.THRESH_BINARY)
ra_m = cv2.dilate(cv2.morphologyEx(ra_bin, cv2.MORPH_OPEN, kernel), kernel, iterations=2)
save("running_avg_morph", ra_m, "RunAvg + Morphology")
ra_c, _ = cv2.findContours(ra_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ra_res = curr_frame.copy()
for c in ra_c:
    if cv2.contourArea(c)<200: continue
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(ra_res, (x,y), (x+w,y+h), (0,255,0), 2)
save("running_avg_result", ra_res, "Running Average 감지 결과")

print(f"\n✅ {step}장 → {OUT_DIR}/")