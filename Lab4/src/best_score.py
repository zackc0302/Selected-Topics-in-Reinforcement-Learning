import os
import re

# 目標資料夾
folder = "log/CarRacing/td3_firstRun"

# 檔名格式：model_36205_131.pth
pattern = re.compile(r"model_(\d+)_(\d+)\.pth")

best_file = None
best_score = -float("inf")

for filename in os.listdir(folder):
    match = pattern.match(filename)
    if match:
        steps, score = map(int, match.groups())
        if score > best_score:
            best_score = score
            best_file = filename

if best_file:
    print(f"最高分的模型是：{best_file}（分數：{best_score}）")
else:
    print("找不到符合格式的檔案。")
