import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 定义类别
class_names = ['healthy', 'rust', 'angular_leaf_spot']

# 加载模型
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# 加载训练好的模型权重，使用 weights_only=True
model.load_state_dict(torch.load('beans_classifier.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()  # 设置为评估模式

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为 PyTorch 张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
])

# 加载并处理图像
def predict_image(image_path):
    # 加载图像并转换为 RGB 格式
    image = Image.open(image_path).convert('RGB')  # 转换为 RGB
    image = transform(image)
    image = image.unsqueeze(0)  # 增加 batch 维度

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]


if __name__ == "__main__":
    image_path = r'D:\Desktop\code\pytorch\bean\test.png'  # 自行替换成自己的图像路径
    prediction = predict_image(image_path)
    print(f'The predicted class for the image is: {prediction}')
