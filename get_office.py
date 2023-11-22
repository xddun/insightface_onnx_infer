import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_sc',
                   providers=['CPUExecutionProvider'])  # 使用的检测模型名为buffalo_sc
app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id小于0表示用cpu预测，det_size表示resize后的图片分辨率
img = cv2.imread("sunyanzi.png")  # 读取图片
faces = app.get(img)  # 得到人脸信息
# print(faces)
for facedata in faces:
    print(facedata["bbox"].shape)  # 人脸框坐标
    print(facedata["kps"].shape)  # 人脸关键点坐标
    print(facedata["det_score"])  # 人脸检测分数
    print(facedata["embedding"].shape)  # 人脸特征向量
