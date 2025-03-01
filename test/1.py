import cv2

# 读取四通道图像（RGBA）
image = cv2.imread("./image.png", cv2.IMREAD_UNCHANGED)

# 检查是否为四通道
if image.shape[2] == 4:
    # 去除 Alpha 通道，保留 RGB
    rgb_image = image[:, :, :3]
else:
    print("图像不是四通道格式")
    rgb_image = image

# 保存或显示结果
cv2.imwrite("output_rgb.jpg", rgb_image)
cv2.imshow("RGB Image", rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()