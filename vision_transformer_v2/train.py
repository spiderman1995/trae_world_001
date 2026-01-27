import os
import math
import argparse # 用于解析命令行参数

import torch
import torch.optim as optim # PyTorch 中的优化器模块。
import torch.optim.lr_scheduler as lr_scheduler # PyTorch 中的学习率调度器模块
from torch.utils.tensorboard import SummaryWriter # 用于记录训练过程中的日志，以便使用 TensorBoard 可视化。
from torchvision import transforms #用于图像预处理


from my_dataset import MyDataSet 
from vit_model import vit_base_patch16_224_in21k as create_model  #默认导入vit_base_patch16模型
from utils import read_split_data, train_one_epoch, evaluate # 工具模块，包含数据分割、训练和评估函数。


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 创建权重保存目录
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter() # 用于记录训练过程中的日志，以便使用 TensorBoard 进行可视化。
    
    # 调用 read_split_data 函数，根据提供的数据集路径 args.data_path，读取数据集并将其分割为训练集和验证集。
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        # 训练集：随机裁剪到 224x224，随机水平翻转，转换为张量，并进行归一化。
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),

        # 验证集：调整大小到 256x256，中心裁剪到 224x224，转换为张量，并进行归一化。
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化 训练 数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化 验证 数据集
    # 使用 MyDataSet 类创建验证数据集实例，传入验证图像路径、标签和预处理流程
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    # 设置数据加载器的参数
    # 设置批量大小 batch_size 和数据加载器的工作线程数 nw，根据 CPU 核心数和批量大小动态调整工作线程数。
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if nw == 0: nw = 1
    print('Using {} dataloader workers every process'.format(nw))

    # 创建训练数据加载器
    # 使用 torch.utils.data.DataLoader 创建训练数据加载器，设置批量大小、随机打乱、内存固定等参数，并传入自定义的 collate_fn。
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               persistent_workers=True,  
                                               prefetch_factor=2,
                                               collate_fn=train_dataset.collate_fn)

    # 创建验证数据加载器
    # 使用 torch.utils.data.DataLoader 创建验证数据加载器，设置批量大小、不随机打乱、内存固定等参数，并传入自定义的 collate_fn。
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             persistent_workers=True,  # 保持工作线程不销毁
                                             prefetch_factor=2,        # 预取因子
                                             collate_fn=val_dataset.collate_fn)
    # 调用 create_model 函数创建 Vision Transformer 模型，指定类别数 args.num_classes 和是否包含额外的 logits 层，将模型移动到指定设备。
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    # 加载预训练权重
    # 如果提供了预训练权重路径 args.weights，加载权重文件并删除不需要的键（如 head 和 pre_logits 层的权重），然后将权重加载到模型中。
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    # 冻结部分权重
    # 如果提供了预训练权重路径 args.weights，加载权重文件并删除不需要的键（如 head 和 pre_logits 层的权重），然后将权重加载到模型中。
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))


    # 创建优化器
    # 筛选出需要梯度更新的参数，使用随机梯度下降（SGD）优化器，设置学习率、动量和权重衰减
    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, 
    #                       lr=args.lr, 
    #                       momentum=0.9, 
    #                       weight_decay=5E-5,
    #                       fused=True)  # 使用 fused 优化器以提高性能

    # 使用SGD 收敛慢且效果通常不如 AdamW 不如一步到位：
    # 对于momentum, AdamW会自动处理, 不需要人为调整,使用默认参数即可
    optimizer = optim.AdamW(pg, 
                            lr=args.lr, 
                            weight_decay=5E-5, 
                            fused=True) # <--- 既换了更强的优化器，又开启了融合
    

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf

    # 创建学习率调度器
    # 定义一个余弦退火学习率调度器，根据训练周期动态调整学习率。
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 训练模型
    
    for epoch in range(args.epochs):
        # train
        # 遍历每个训练周期，调用 train_one_epoch 函数进行训练，记录训练损失和准确率。
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        # 更新学习率
        # 调用学习率调度器的 step 方法，更新优化器的学习率。
        scheduler.step()

        # validate
        # 验证模型
        # 调用 evaluate 函数对验证集进行评估，记录验证损失和准确率
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        # 记录日志
        # 使用 SummaryWriter 记录训练和验证的损失、准确率以及学习率，以便后续使用 TensorBoard 可视化。
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        # 保存模型权重
        # 每个训练周期结束后，将模型的权重保存到 ./weights 目录下。
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

# 解析命令行参数并调用主函数
# 包括类别数、训练周期、批量大小、学习率等。
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
