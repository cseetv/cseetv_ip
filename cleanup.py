import json, os

with open("data/nightowls_validation.json", encoding="utf-8") as f:
    data = json.load(f)

# V001 파일명 목록
keep = set(
    img["file_name"] for img in data["images"]
    if 7000000 <= img["id"] <= 7001317
)
print(f"유지할 파일: {len(keep)}개")

# 나머지 삭제
deleted = 0
folder = "data/nightowls_images"
for fname in os.listdir(folder):
    if fname not in keep:
        os.remove(os.path.join(folder, fname))
        deleted += 1
        if deleted % 5000 == 0:
            print(f"  삭제 중... {deleted}개")

print(f"삭제 완료: {deleted}개 (V001 {len(keep)}개 유지)")