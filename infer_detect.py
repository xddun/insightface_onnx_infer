import cv2

from retinaface import RetinaFace

detector = RetinaFace(model_file="buffalo_sc/det_500m.onnx", providers=['CPUExecutionProvider'])
detector.prepare(0, input_size=(640, 640), nms_thresh=0.4, det_thresh=0.5)

img = cv2.imread("sunyanzi.png")
det, kpss = detector.detect(img)
print(type(det), det)  # 左上角和右下角坐标+置信度
print(type(kpss), kpss)

# 绘制框和关键点
for i in range(det.shape[0]):
    cv2.rectangle(img, (int(det[i][0]), int(det[i][1])), (int(det[i][2]), int(det[i][3])), (0, 255, 0), 2)
    for j in range(kpss.shape[1]):
        cv2.circle(img, (int(kpss[i][j][0]), int(kpss[i][j][1])), 2, (0, 0, 255), 2)

cv2.imwrite("sunyanzi_retinaface.png", img)
