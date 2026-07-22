import cv2
import numpy as np
import os


def compute_mean_std(image_paths):
    # 初始化变量
    mean = np.zeros(3)
    std = np.zeros(3)
    pixel_num = 0

    for path in image_paths:
        img = cv2.imread(path)  # BGR
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 可选，转换为RGB

        # 更新均值
        mean += np.sum(img, axis=(0, 1))
        pixel_num += img.shape[0] * img.shape[1]

    # 计算均值
    mean /= pixel_num

    # 第二次遍历计算标准差
    for path in image_paths:
        img = cv2.imread(path)  # BGR
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 可选，转换为RGB

        std += np.sum((img - mean) ** 2, axis=(0, 1))

    std = np.sqrt(std / pixel_num)

    return mean, std


def compute_ycrcb_mean_std(image_paths):
    # 初始化变量
    mean = np.zeros(3)
    std = np.zeros(3)
    pixel_num = 0

    for path in image_paths:
        img = cv2.imread(path)  # BGR
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # 转换为 YCrCb

        # 更新均值
        mean += np.sum(img_ycrcb, axis=(0, 1))
        pixel_num += img_ycrcb.shape[0] * img_ycrcb.shape[1]

    # 计算均值
    mean /= pixel_num

    # 第二次遍历计算标准差
    for path in image_paths:
        img = cv2.imread(path)  # BGR
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # 转换为 YCrCb

        std += np.sum((img_ycrcb - mean) ** 2, axis=(0, 1))

    std = np.sqrt(std / pixel_num)

    return mean, std


# 获取训练集图像路径
# EMDS7
emds7_path = [os.path.join('datasets/EMDS7/EMDS7Images/trainval', f)  # 1909 images
              for f in os.listdir('datasets/EMDS7/EMDS7Images/trainval')
              # if f.endswith('.png')
              ]

# 范围测试
# test_image = np.array([
#     [[0, 0, 0],         # 黑
#      [255, 255, 255],   # 白
#      [255, 0, 0],       # 蓝
#      [0, 255, 0],       # 绿
#      [0, 0, 255],       # 红
#      [255, 255, 0],     # 青
#      [255, 0, 255],     # 洋红
#      [0, 255, 255]],    # 黄
# ], dtype=np.uint8)
#
# ycrcb = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCrCb)
#
# print("YCrCb 值范围:")
# print("Y  范围:", ycrcb[:, :, 0].min(), "-", ycrcb[:, :, 0].max())
# print("Cr 范围:", ycrcb[:, :, 1].min(), "-", ycrcb[:, :, 1].max())
# print("Cb 范围:", ycrcb[:, :, 2].min(), "-", ycrcb[:, :, 2].max())
"""
YCrCb 值范围:
Y  范围: 0 - 255
Cr 范围: 0 - 255
Cb 范围: 1 - 255
"""

# 计算均值和标准差
mean_bgr, std_bgr = compute_mean_std(emds7_path)

print("\nBGR 空间:")
print(f"Mean [0,255]: {mean_bgr}")
print(f"Std [0,255]: {std_bgr}")
print(f"Mean [0,1]: {mean_bgr / 255.0}")
print(f"Std [0,1]: {std_bgr / 255.0}")

mean_ycrcb, std_ycrcb = compute_ycrcb_mean_std(emds7_path)

print("YCrCb 空间:")
print(f"Mean [0,255]: {mean_ycrcb}")
print(f"Std [0,255]: {std_ycrcb}")
print(f"Mean [0,1]: {mean_ycrcb / 255.0}")
print(f"Std [0,1]: {std_ycrcb / 255.0}")


"""
EMDS7
BGR 空间:
Mean [0,255]: [ 86.14748405 173.19292289 116.08142144]
Std [0,255]: [20.43483811 37.45326237 24.48730807]
Mean [0,1]: [0.33783327 0.67918793 0.45522126]
Std [0,1]: [0.08013662 0.14687554 0.09602866]
YCrCb 空间:
Mean [0,255]: [146.18436032 106.49779555  94.13277945]
Std [0,255]: [30.51703079  8.24909203 10.8226398 ]
Mean [0,1]: [0.573272   0.41763841 0.36914815]
Std [0,1]: [0.11967463 0.03234938 0.04244172]
"""
