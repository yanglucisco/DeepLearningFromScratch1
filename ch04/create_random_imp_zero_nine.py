from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

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

# 显示
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap="gray")
    ax.set_title(f"Digit {i}")
    ax.axis("off")

plt.tight_layout()
plt.show()
