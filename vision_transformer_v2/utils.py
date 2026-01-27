import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

# 图像分类任务的工具函数集合 包括数据读取、数据可视化、模型训练和验证等
# 数据读取与划分 (read_split_data)，读取指定目录下的图像数据，并按照一定比例划分训练集和验证集。
def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别  ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引 {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    # class_indices.items() 返回一个可迭代对象，其中包含了字典中的所有键值对（key-value pairs）
    # (val, key) 创建了一个新的元组，将键和值的位置进行了互换，得到 (0, 'daisy')
    # json.dumps() 将 Python 对象（如字典、列表等）转换为一个 JSON 格式的字符串。
    # {
    # "0": "daisy",
    # "1": "dandelion",
    # "2": "roses",
    # "3": "sunflowers",
    # "4": "tulips"
    # }
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储 训练集的  所有图片路径
    train_images_label = []  # 存储 训练集 图片 对应索引信息
    val_images_path = []  # 存储 验证集的 所有图片路径
    val_images_label = []  # 存储 验证集 图片 对应索引信息
    every_class_num = []  # 存储 每个类别 的 样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径，os.path.splitext(i)[-1] 把 一个文件名或路径拆分成 文件名 和 扩展名 两部分 。-1 取最后一个元素
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        # random.sample(population, k) population：要从中抽样的序列，这里是 images 列表，它包含了所有图像的路径。
        # Python random 模块中的一个函数，用于从一个序列（如列表）中随机、无放回地选择 k 个元素
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label

# 从一个 PyTorch DataLoader 中读取一个批次（batch）的数据，然后将其中的几张图片连同它们的标签一起可视化显示出来 。代码其他地方没有使用到
def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()

# 使用 pickle 库将一个列表（list_info）对象，以二进制的形式序列化并保存到指定的文件（file_name）中。
# pickle 模块可以处理几乎所有的 Python 对象，不仅仅是列表。
# pickle.dump(obj, file) 的作用是：将 Python 对象 obj 序列化（也称为 “腌制”）成一个字节流，并将这个字节流写入到打开的文件对象 file 中。
# 代码中未使用到
def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    # model: 待训练的PyTorch模型
    # data_loader: DataLoader对象，提供训练数据的迭代器
    # device: 训练设备（'cuda'或'cpu'）

    model.train()
    # CrossEntropyLoss适用于多分类问题，结合了log_softmax和NLLLoss
    loss_function = torch.nn.CrossEntropyLoss()
    # 初始化累计损失和累计正确预测数
    # 使用torch.zeros(1).to(device)确保张量在正确的设备上
    accu_loss = torch.zeros(1).to(device)  # 累计损失，# 累计整个epoch的损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    # 4. 清空优化器的梯度
    # 防止上一轮迭代的梯度影响当前迭代
    optimizer.zero_grad()

    # 5. 初始化样本计数器
    sample_num = 0
    # 6. 使用tqdm包装DataLoader，创建进度条
    data_loader = tqdm(data_loader, file=sys.stdout)
     # 7. 遍历DataLoader中的每个batch
    for step, data in enumerate(data_loader):
        # 7.1.
        images, labels = data

        # 7.2.累加当前batch的样本数
        # images.shape[0]是当前batch的大小
        sample_num += images.shape[0]

        # 7.3. 前向传播：计算模型预测结果
        # 将图像数据移动到指定设备，然后输入模型
        pred = model(images.to(device))

        # 7.4. 获取预测的类别
        # torch.max(pred, dim=1)返回每一行的最大值和对应的索引
        #  [1]表示只取索引（即预测的类别）
        pred_classes = torch.max(pred, dim=1)[1]

        # 7.5. 计算并累加正确预测的样本数
        # torch.eq比较预测类别与真实标签是否相等，返回布尔张量
        # .sum()将布尔张量转换为整数并求和
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # 7.6. 计算损失
        # loss_function的输入是模型的原始输出(pred)和真实标签(labels)
        loss = loss_function(pred, labels.to(device))

        # 7.7. 反向传播：计算梯度
        loss.backward()

        # 7.8. 累加损失（detach()避免计算梯度）
        # .detach()将loss从计算图中分离出来，只取其数值
        accu_loss += loss.detach()

        # 7.9. 更新进度条描述信息
        # 计算当前平均损失和平均准确率并显示
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        # 7.10. 检查损失是否为无穷大或NaN
        # 如果损失异常，停止训练并报错
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        # 7.11. 优化器更新模型参数
        # 根据计算出的梯度调整模型权重
        optimizer.step()

        # 7.12. 清空梯度，为下一个batch做准备
        optimizer.zero_grad()

    # 8. 计算并返回整个epoch的平均损失和平均准确率
    # .item()将张量转换为Python标量
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

# 用于在验证集（或测试集）上评估模型的性能，它不涉及模型参数的更新。

# 装饰器的作用。 一个上下文管理器和函数装饰器。在代码块执行期间，禁用梯度计算。
# 禁用梯度计算可以带来两个显著好处：节省内存、加速计算
@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    # 1. 定义损失函数（与训练时一致）
    # 用于计算验证集上的损失，以便监控模型是否过拟合
    loss_function = torch.nn.CrossEntropyLoss()

    # 2. 将模型设置为评估模式
    # 这会禁用dropout、冻结batch normalization的统计量等，确保评估结果准确
    model.eval()

    # 3. 初始化累计损失和累计正确预测数
    # 使用torch.zeros(1).to(device)确保张量在正确的设备上
    accu_num = torch.zeros(1).to(device)   # 累计整个验证集预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计整个验证集损失

    # 4. 初始化样本计数器
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        # 前向传播,由于在@torch.no_grad()上下文下，不会计算梯度，节省内存和计算资源
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        # 更新进度条描述信息
        # 计算当前平均损失和平均准确率并显示
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    # 7. 计算并返回整个验证集的平均损失和平均准确率
    # .item()将张量转换为Python标量
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
