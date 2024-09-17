import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import os
import json
from datetime import datetime
from model import MultiModalModel
from torch.amp import GradScaler, autocast

torch.manual_seed(42)
np.random.seed(42)

class Config:
    num_epochs = 45
    batch_size = 16
    accumulation_steps = 1
    learning_rate = 0.001
    num_classes = 155
    num_points = 17
    log_interval = 10
    save_interval = 5
    val_split = 0.3
    experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

config = Config()

# 设置日志
def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

# 数据加载
def load_data(file_path):
    return np.load(file_path, mmap_mode='r')

# 数据预处理
def preprocess_data(data, is_label=False):
    if is_label:
        return torch.from_numpy(data.copy()).long()
    else:
        return torch.from_numpy(data.copy()).float()

def main():
    os.makedirs(f"logs/{config.experiment_name}", exist_ok=True)
    os.makedirs(f"models/{config.experiment_name}", exist_ok=True)
    train_logger = setup_logger('train_logger', f'logs/{config.experiment_name}/train.log')
    val_logger = setup_logger('val_logger', f'logs/{config.experiment_name}/val.log')

    # TensorBoard 设置
    writer = SummaryWriter(f'runs/{config.experiment_name}')

    # 定义图结构
    graph = [
        (10, 8), (8, 6), (9, 7), (7, 5), (15, 13), (13, 11), (16, 14), (14, 12),
        (11, 5), (12, 6), (11, 12), (5, 6), (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2)
    ]

    # 创建邻接矩阵
    A = np.zeros((config.num_points, config.num_points))
    for i, j in graph:
        A[j, i] = A[i, j] = 1
    A = np.stack([A])

    # 数据加载和预处理
    train_joint = preprocess_data(load_data('data/train_joint.npy'))
    train_bone = preprocess_data(load_data('data/train_bone.npy'))
    train_motion = preprocess_data(load_data('data/train_bone_motion.npy'))
    train_labels = preprocess_data(load_data('data/train_label.npy'), is_label=True)

    # 创建完整的训练数据集
    full_train_dataset = TensorDataset(train_joint, train_bone, train_motion, train_labels)

    # 分割训练集和验证集
    train_size = int((1 - config.val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # 初始化模型、损失函数和优化器
    num_person = train_joint.shape[4]  # 获取数据中的人数
    model = MultiModalModel(num_class=config.num_classes, num_point=config.num_points,
                            num_person=num_person, graph=A, in_channels=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"使用设备: {device}")
    print(f"模型参数: num_class={config.num_classes}, num_point={config.num_points}, num_person={num_person}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    scaler = GradScaler(enabled=(device.type != 'cpu'))

    # 训练函数
    def train_epoch(epoch):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}')
        for i, (batch_joint, batch_bone, batch_motion, batch_labels) in enumerate(pbar):
            batch_joint, batch_bone, batch_motion, batch_labels = [b.to(device) for b in
                                                                   [batch_joint, batch_bone, batch_motion, batch_labels]]

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(batch_joint, batch_bone, batch_motion)
                loss = criterion(outputs, batch_labels)

            scaler.scale(loss).backward()

            if (i + 1) % config.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_labels.size(0)
            train_correct += predicted.eq(batch_labels).sum().item()

            if i % config.log_interval == 0:
                train_logger.info(f'Epoch [{epoch + 1}/{config.num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                                  f'Loss: {loss.item():.4f}, Acc: {100. * train_correct / train_total:.2f}%')

            pbar.set_postfix(
                {'loss': f'{train_loss / (i+1):.4f}', 'acc': f'{100. * train_correct / train_total:.2f}%'})

        return train_loss / len(train_loader), 100. * train_correct / train_total

    # 验证函数
    def validate(epoch):
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_joint, batch_bone, batch_motion, batch_labels in val_loader:
                batch_joint, batch_bone, batch_motion, batch_labels = [b.to(device) for b in
                                                                       [batch_joint, batch_bone, batch_motion,
                                                                        batch_labels]]
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(batch_joint, batch_bone, batch_motion)
                    loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()

                all_predictions.extend(outputs.softmax(dim=1).cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        val_accuracy = 100. * val_correct / val_total
        val_loss /= len(val_loader)

        val_logger.info(f'Epoch [{epoch + 1}/{config.num_epochs}], '
                        f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

        return val_loss, val_accuracy, np.array(all_predictions), np.array(all_labels)

    # 保存训练记录
    def save_training_record(epoch, train_loss, train_acc, val_loss, val_acc):
        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        with open(f'logs/{config.experiment_name}/training_record.json', 'a') as f:
            json.dump(record, f)
            f.write('\n')

    # 主训练循环
    best_val_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(config.num_epochs):
        train_loss, train_acc = train_epoch(epoch)
        val_loss, val_acc, predictions, labels = validate(epoch)

        # 更新 history 字典
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        save_training_record(epoch, train_loss, train_acc, val_loss, val_acc)

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'models/{config.experiment_name}/best_model.pth')

            # 保存置信度文件
            np.save(f'logs/{config.experiment_name}/best_predictions.npy', predictions)
            np.save(f'logs/{config.experiment_name}/true_labels.npy', labels)

        if (epoch + 1) % config.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, f'models/{config.experiment_name}/checkpoint_{epoch + 1}.pth')

    # 保存最终模型
    torch.save(model.state_dict(), f'models/{config.experiment_name}/final_model.pth')

    # 绘制训练历史图表
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'logs/{config.experiment_name}/training_history.png')
    plt.close()

    writer.close()
    print(f'训练完成。所有日志和模型都保存在 logs/{config.experiment_name} 和 models/{config.experiment_name} 目录下。')

if __name__ == '__main__':
    main()