# coding: utf-8
import sys, os
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

def create_image():
    images = []

    for i in range(10):
        img = Image.new("L", (28, 28), color=0)
        draw = ImageDraw.Draw(img)
        num = str(i)

        # 使用默认字体（Linux/Mac）
        font = ImageFont.load_default()

        bbox = font.getbbox(num)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((28 - w) / 2, (28 - h) / 2 - 2), num, fill=255, font=font)

        images.append(np.array(img))

    images = np.stack(images)
    return images

# 加载测试数据
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
imgs = create_image()
x_test = np.random.randint(0, 256, size=(10, 784), dtype=np.uint8)
t_test = np.random.randint(0, 256, size=(10, 10), dtype=np.uint8)
# 加载训练好的参数
with open("ch04/network_params.pkl", "rb") as f:
    params = pickle.load(f)

W1, W2 = params['W1'], params['W2']
b1, b2 = params['b1'], params['b2']

def predict(img):
    a1 = np.dot(img, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)
    return np.argmax(y), np.max(y)  # 返回预测数字和置信度

# 测试前10张图片
print("=" * 40)
print("测试前10张图片的识别结果：")
print("=" * 40)
for i in range(10):
    pred, confidence = predict(x_test[i])
    true_label = np.argmax(t_test[i])
    mark = "✓" if pred == true_label else "✗"
    print(f"图片{i:2d}  预测={pred}  真实={true_label}  置信度={confidence:.4f}  {mark}")

# 整体准确率
correct = 0
for i in range(len(x_test)):
    pred, _ = predict(x_test[i])
    if pred == np.argmax(t_test[i]):
        correct += 1

print("=" * 40)
print(f"测试集整体准确率: {correct / len(x_test) * 100:.2f}%")
print("=" * 40)
