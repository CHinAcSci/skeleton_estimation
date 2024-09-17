import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import MultiModalModel
import os
from tqdm import tqdm
import logging
from datetime import datetime

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

def load_data(file_path):
    return np.load(file_path)

def preprocess_data(data, is_label=False):
    if is_label:
        return torch.from_numpy(data.copy()).long()
    else:
        return torch.from_numpy(data.copy()).float()

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_joint, batch_bone, batch_motion, batch_labels in tqdm(test_loader, desc="Evaluating"):
            batch_joint, batch_bone, batch_motion, batch_labels = [b.to(device) for b in
                                                                   [batch_joint, batch_bone, batch_motion,
                                                                    batch_labels]]

            outputs = model(batch_joint, batch_bone, batch_motion)
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()

            all_predictions.extend(outputs.softmax(dim=1).cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    accuracy = 100. * correct / total
    return accuracy, np.array(all_predictions), np.array(all_labels)

def main():
    log_dir = "logs"
    setup_logger(log_dir)

    # 设置参数
    model_path = "E:/Skeleton_estimation/Skeleton_estimation/models/experiment_20240915_002537/best_model.pth"
    test_joint_path = "data/test_A_joint.npy"
    test_bone_path = "data/test_A_bone.npy"
    test_motion_path = "data/test_A_bone_motion.npy"
    test_label_path = "data/test_A_label.npy"
    output_dir = "evaluation_results"
    batch_size = 16
    num_classes = 155
    num_points = 17

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # 加载测试数据
    logging.info("Loading test data")
    test_joint = preprocess_data(load_data(test_joint_path))
    test_bone = preprocess_data(load_data(test_bone_path))
    test_motion = preprocess_data(load_data(test_motion_path))
    test_labels = preprocess_data(load_data(test_label_path), is_label=True)
    logging.info("Test data loaded successfully.")

    # 测试数据集和数据加载器
    test_dataset = TensorDataset(test_joint, test_bone, test_motion, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    logging.info(f"Test DataLoader created with batch size: {batch_size}")

    # 图结构
    graph = [
        (10, 8), (8, 6), (9, 7), (7, 5), (15, 13), (13, 11), (16, 14), (14, 12),
        (11, 5), (12, 6), (11, 12), (5, 6), (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2)
    ]
    A = np.zeros((num_points, num_points))
    for i, j in graph:
        A[j, i] = A[i, j] = 1
    A = np.stack([A])

    # 初始化模型
    num_person = test_joint.shape[4]
    model = MultiModalModel(num_class=num_classes, num_point=num_points,
                            num_person=num_person, graph=A, in_channels=3)
    model = model.to(device)
    logging.info("Model initialized and moved to device.")

    # 加载模型
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    logging.info(f"Model loaded from: {model_path}")

    # 评估模型
    logging.info("Starting model evaluation...")
    accuracy, predictions, true_labels = evaluate_model(model, test_loader, device)

    logging.info(f"Test Accuracy: {accuracy:.2f}%")

    # 保存预测结果和真实标签
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'test_predictions.npy'), predictions)
    np.save(os.path.join(output_dir, 'test_true_labels.npy'), true_labels)
    logging.info(f"Predictions and true labels saved in: {output_dir}")

if __name__ == "__main__":
    main()