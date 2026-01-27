import os
import json

import torch # PyTorch 深度学习框架。
from PIL import Image # 用于图像处理。
from torchvision import transforms # 用于图像预处理。
import matplotlib.pyplot as plt # 用于绘图

from vit_model import vit_base_patch16_224_in21k as create_model


def main():
    # 检查是否有可用的 GPU，如果有，则使用 GPU (cuda:0)，否则使用 CPU。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),   # 将图像缩放到 256x256。
         transforms.CenterCrop(224), # 从中心裁剪出 224x224 的图像。
         transforms.ToTensor(), # 将图像转换为 PyTorch 张量
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) # 对图像进行归一化处理，均值和标准差均为 0.5

    # load image
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)  # 使用 matplotlib.pyplot.imshow 显示图像。
    # [N, C, H, W]
    img = data_transform(img) # 预处理
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0) # 在第 0 维（批量维度）添加一个维度，将图像从 [C, H, W] 转换为 [1, C, H, W]，以符合模型输入的要求。

    # read class_indict 读取类别索引文件
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model  create_model 函数创建 Vision Transformer 模型，指定类别数为 5，has_logits=False 表示不使用额外的预训练层。
    model = create_model(num_classes=5, has_logits=False).to(device)  # 如果在训练的时候has_logits设置成true，这里预测也要设置成true
    # load model weights
    model_weight_path = "./weights/model-9.pth"  # 指定模型权重文件路径
    # torch.load ，用于加载保存的 PyTorch 对象（如模型权重、张量、优化器状态等
    # model.load_state_dict 用于将保存的模型权重加载到一个模型实例中
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval() # 将模型设置为评估模式（model.eval()），关闭 Dropout 等训练时的特性
    with torch.no_grad(): # 禁用梯度计算，减少内存占用。
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu() # 去掉批量维度。
        predict = torch.softmax(output, dim=0) # 计算预测概率。
        predict_cla = torch.argmax(predict).numpy() # 获取预测类别索引。
    # 格式化预测结果字符串。
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    # 将预测结果设置为图像的标题
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
