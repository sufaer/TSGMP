from torch.utils.data import Dataset
from PIL import Image
import os
import sys
import json
import pickle
import random
import torch
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np 
import torch.nn.functional as F
from tqdm import tqdm
from pandas import Series, DataFrame
from sklearn.metrics import roc_auc_score
import logging
import glob
import torchvision.transforms.functional as F


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./log/dataset.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)



def read_split_data_by_hospital(
    csv_file_path: str,
    image_root: str,
    train_hospitals: list = [3, 5],
    random_seed: int = 0
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    assert os.path.exists(csv_file_path), f"CSV file: {csv_file_path} does not exist."
    assert os.path.exists(image_root), f"Image root directory: {image_root} does not exist."

    df = pd.read_csv(csv_file_path, encoding='gbk')

    if 'hospital' not in df.columns:
        raise ValueError("CSV file must contain a 'hospital' column.")
    if 'patient_ID' not in df.columns:
        raise ValueError("CSV file must contain a 'patient_ID' column.")
    if 'bpCR' not in df.columns:
        raise ValueError("CSV file must contain a 'bpCR' column.")

    df['patient_ID'] = df['patient_ID'].astype(str)
    df['bpCR'] = df['bpCR'].astype(int)
    df['hospital'] = df['hospital'].astype(int) # 确保 hospital 列是整数类型

    class_labels = sorted(df['bpCR'].unique().tolist())
    class_indices = dict((k, v) for v, k in enumerate(class_labels))
    
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    logger.info(f"Class indices: {json_str}")

    all_patient_info = [] # 将存储包含所有图像路径的字典

    # 遍历每个患者ID，收集其所有模态图像路径
    for patient_id, group_df in df.groupby('patient_ID'):
        # 获取患者级别的信息 (假设在同一patient_ID下标签和医院ID是唯一的)
        label = group_df['bpCR'].iloc[0]
        hospital_id = group_df['hospital'].iloc[0]

        # 根据命名约定构建图像路径
        pre_dwi = os.path.join(image_root, f"{patient_id}_dwi1.png")
        pre_dce = os.path.join(image_root, f"{patient_id}_c1.png")
        post_dwi = os.path.join(image_root, f"{patient_id}_dwi2.png")
        post_dce = os.path.join(image_root, f"{patient_id}_c2.png")

        # 只有当一个患者的所有三个图像都存在时，才将其包含在数据集中
        if os.path.exists(pre_dwi) and os.path.exists(pre_dce) and os.path.exists(post_dwi) and os.path.exists(post_dce):
            all_patient_info.append({
                'patient_id': patient_id,
                'label': label,
                'hospital': hospital_id,
                'pre_dwi': pre_dwi,
                'pre_dce': pre_dce,
                'post_dwi': post_dwi,
                'post_dce': post_dce
            })
        else:
            missing_images = []
            if not os.path.exists(pre_dwi): missing_images.append(f"{patient_id}_dwi1.png")
            if not os.path.exists(pre_dce): missing_images.append(f"{patient_id}_c1.png")
            if not os.path.exists(post_dwi): missing_images.append(f"{patient_id}_dwi2.png")
            if not os.path.exists(post_dce): missing_images.append(f"{patient_id}_c2.png")
            logger.info(f"Warning: Missing one or more MRI sequences for patient {patient_id}. Skipping patient. Missing files: {', '.join(missing_images)}")

    if not all_patient_info:
        raise ValueError("No complete patient data (DWI, CE-T1WI, T2WI) found based on the provided CSV and image root.")

    all_patient_df = pd.DataFrame(all_patient_info)

    # 训练集和验证集（来自训练医院）
    train_val_data = all_patient_df[all_patient_df['hospital'].isin(train_hospitals)]
    
    # 外部测试集 (所有不在训练医院列表中的医院)
    external_test_data = all_patient_df[~all_patient_df['hospital'].isin(train_hospitals)]
    
    dict_of_test_sets = {}
    test_hospital_ids = sorted(external_test_data['hospital'].unique().tolist())
    
    for hospital_id in test_hospital_ids:
        hospital_data = external_test_data[external_test_data['hospital'] == hospital_id].to_dict('records')
        dict_of_test_sets[hospital_id] = hospital_data

    # 打印划分统计信息
    logger.info(f"\n--- Dataset Split Statistics ---")
    logger.info(f"Total complete patient MRI sequences found: {len(all_patient_info)}")
    logger.info(f"Train patients: {len(train_val_data)} (Hospitals: {train_hospitals})")
    
    logger.info("\n--- External Test Set Details ---")
    if not dict_of_test_sets:
        logger.info("No external test sets defined.")
    else:
        for hospital_id, data_list in dict_of_test_sets.items():
            logger.info(f"  Hospital {hospital_id}: {len(data_list)} patients")
            if data_list:
                labels = [d['label'] for d in data_list]
                logger.info(f"    Class Distribution: {pd.Series(labels).value_counts().sort_index().rename(index=class_indices).to_dict()}")

    return train_val_data.to_dict('records'), dict_of_test_sets


class MyDataSet(Dataset):
    """
    一个用于加载 MRI 图像数据集的自定义数据集类。
    支持加载 DWI, CE-T1WI, T2WI 三种模态的图像，并返回相应的张量、标签和图像名称。
    """
    def __init__(self, patient_data_list, transform=None):
        """
        Args:
            patient_data_list (list of dict): 包含患者信息的字典列表。
                                              每个字典应包含 'dwi_path', 'cet1wi_path', 't2wi_path', 'label', 'patient_id'。
            transform (callable, optional): 应用于每个图像的转换。
        """
        self.patient_data_list = patient_data_list
        self.transform = transform

    def __len__(self):
        return len(self.patient_data_list)

    def __getitem__(self, item):
        patient_info = self.patient_data_list[item]

        pre_dwi_path = patient_info['pre_dwi']
        pre_dce_path = patient_info['pre_dce']
        post_dwi_path = patient_info['post_dwi']
        post_dce_path = patient_info['post_dce']
        label = patient_info['label']
        patient_id = patient_info['patient_id']

        # 加载并预处理 DWI 图像
        # 确保图像以 RGB 格式加载，以适配预训练模型
        # 加载图像并转换为灰度图
        # 注意：这里需要确保transform能够处理PIL Image对象，或者进行必要的to_tensor转换
        pre_dwi = Image.open(pre_dwi_path).convert('L')
        pre_dce = Image.open(pre_dce_path).convert('L')
        post_dwi = Image.open(post_dwi_path).convert('L')
        post_dce = Image.open(post_dce_path).convert('L')

        # 如果有transform，则对所有图像应用相同的transform
        # if self.transform is not None:
        #     pre_dwi = self.transform(pre_dwi)
        #     pre_dce = self.transform(pre_dce)
        #     post_dwi = self.transform(post_dwi)
        #     post_dce = self.transform(post_dce)

        if self.transform is not None:
            # 判断 transform 的类型来选择正确的处理方式
            if isinstance(self.transform, PairedTransforms):
                # 训练阶段：使用 PairedTransforms，对所有四张图像应用相同随机变换
                pre_dwi, post_dwi, pre_dce, post_dce = self.transform(
                    pre_dwi, post_dwi, pre_dce, post_dce
                )
            elif isinstance(self.transform, transforms.Compose):
                # 测试阶段：使用 Compose，对每张图像分别应用确定性变换
                pre_dwi = self.transform(pre_dwi)
                post_dwi = self.transform(post_dwi)
                pre_dce = self.transform(pre_dce)
                post_dce = self.transform(post_dce)
            else:
                raise TypeError("Unsupported transform type. Please use either PairedTransforms or transforms.Compose.")

            
        return pre_dwi, pre_dce, post_dwi, post_dce, label, patient_id

    @staticmethod
    def collate_fn(batch):
        """
        自定义 collate_fn，用于将多个样本组成一个批次。
        这个函数需要适应新的返回格式 (dwi_image, cet1wi_image, t2wi_image, label, patient_id)。
        """
        pre_dwi, pre_dce, post_dwi, post_dce, labels, patient_ids = zip(*batch)

        # 将图像张量堆叠成批次
        pre_dwi = torch.stack(pre_dwi, 0)
        pre_dce = torch.stack(pre_dce, 0)
        post_dwi = torch.stack(post_dwi, 0)
        post_dce = torch.stack(post_dce, 0)

        # 将标签转换为 Tensor
        labels_batch = torch.as_tensor(labels, dtype=torch.long)

        return pre_dwi, pre_dce, post_dwi, post_dce, labels_batch, patient_ids
     



# 保存交叉验证的train和val数据集
def save_fold_splits(fold, train_idx, val_idx, train_data, save_dir="cv_splits"):
    """
    保存每折的训练集和验证集划分
    
    参数:
        fold: 当前折数 (0-based)
        train_idx: 训练集索引
        val_idx: 验证集索引 
        train_data: 原始训练数据列表
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取患者信息
    train_patients = [train_data[i]['patient_id'] for i in train_idx]
    val_patients = [train_data[i]['patient_id'] for i in val_idx]
    
    # 创建DataFrame
    df = pd.DataFrame({
        'patient_id': train_patients + val_patients,
        'fold': fold + 1,
        'split': ['train'] * len(train_patients) + ['val'] * len(val_patients),
        'label': [train_data[i]['label'] for i in train_idx] + 
                [train_data[i]['label'] for i in val_idx]
    })
    
    # 保存到CSV
    save_path = os.path.join(save_dir, f"fold_{fold+1}_split.csv")
    df.to_csv(save_path, index=False)
    logger.info(f"Fold {fold+1} 划分已保存到 {save_path}")
    
    return df


def first_plot_losses(train_losses, test_losses, fold, save_dir):
    plt.figure()
    plt.plot(train_losses, label='Training Tatol Loss')
    plt.plot(test_losses, label='Validation Tatol Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Fold {fold + 1} Loss Curve')
    plt.savefig(os.path.join(save_dir, f'first_fold_{fold + 1}_loss_curve.png'))
    plt.close()


# def get_train_transforms(img_size=224):
#     """
#     获取推理阶段的图像预处理变换。
#     """
#     normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    
#     preprocess = transforms.Compose([
#             # ----- 建议添加的更多数据扩增 -----
#             transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(3./4, 4./3)),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(), # 随机垂直翻转
#             transforms.RandomRotation(degrees=15), # 随机旋转，例如 -15 到 +15 度
#             # RandomAffine可以包含平移、缩放、旋转、剪切，非常强大
#             transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0),
#             # ------------------------------------
#             transforms.ToTensor(),
#             normalize
#         ])

#     return preprocess


def get_train_transforms(img_size=224):
    """
    获取训练阶段的图像预处理变换，确保成对图像（pre/post）应用相同的随机变换。
    """
    # 训练阶段的随机变换列表
    train_transforms = [
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(3./4, 4./3)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0)
    ]
    
    # 测试阶段的确定性变换
    # 这部分变换通常在数据增强后应用，以确保图像尺寸和归一化的一致性
    final_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    return PairedTransforms(train_transforms, final_transforms)



def get_test_transforms(img_size=224):
    """
    获取测试阶段的图像预处理变换，不包含随机性。
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


class PairedTransforms:
    def __init__(self, random_transforms, final_transforms):
        self.random_transforms = random_transforms
        self.final_transforms = final_transforms

    def __call__(self, img_pre_dwi, img_post_dwi, img_pre_dce, img_post_dce):
        # 设置随机种子
        seed = random.randint(0, 100000)
        
        # 特殊处理 RandomResizedCrop
        transform = self.random_transforms[0]
        i, j, h, w = transform.get_params(img_pre_dwi, transform.scale, transform.ratio)
        img_pre_dwi = F.resized_crop(img_pre_dwi, i, j, h, w, transform.size, transform.interpolation)
        img_post_dwi = F.resized_crop(img_post_dwi, i, j, h, w, transform.size, transform.interpolation)
        img_pre_dce = F.resized_crop(img_pre_dce, i, j, h, w, transform.size, transform.interpolation)
        img_post_dce = F.resized_crop(img_post_dce, i, j, h, w, transform.size, transform.interpolation)

        # 循环处理其他随机变换
        for transform in self.random_transforms[1:]:
            torch.manual_seed(seed)
            img_pre_dwi = transform(img_pre_dwi)
            torch.manual_seed(seed)
            img_post_dwi = transform(img_post_dwi)
            torch.manual_seed(seed)
            img_pre_dce = transform(img_pre_dce)
            torch.manual_seed(seed)
            img_post_dce = transform(img_post_dce)

        # 应用确定性变换
        img_pre_dwi = self.final_transforms(img_pre_dwi)
        img_post_dwi = self.final_transforms(img_post_dwi)
        img_pre_dce = self.final_transforms(img_pre_dce)
        img_post_dce = self.final_transforms(img_post_dce)
        
        return img_pre_dwi, img_post_dwi, img_pre_dce, img_post_dce


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()
    data_loader = tqdm(data_loader, file=sys.stdout)
    step = 0
    for pre_dwi, pre_dce, post_dwi, post_dce, labels_batch, patient_ids in data_loader:
        pre_dwi = pre_dwi.to(device)
        pre_dce = pre_dce.to(device)
        post_dwi = post_dwi.to(device)
        post_dce = post_dce.to(device)
        labels = labels_batch.to(device)
        loss = model(pre_dwi, pre_dce, post_dwi, post_dce, labels)
        loss.backward()
        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            logger.info('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1)


def valid_one_epoch(model, data_loader, device, epoch):
    model.eval() # 设置模型为评估模式
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    data_loader = tqdm(data_loader, file=sys.stdout) # 使用 tqdm 包装数据加载器
    step = 0
    with torch.no_grad(): # 在评估模式下禁用梯度计算
        for pre_dwi, pre_dce, post_dwi, post_dce, labels_batch, patient_ids in data_loader:
            pre_dwi = pre_dwi.to(device)
            pre_dce = pre_dce.to(device)
            post_dwi = post_dwi.to(device)
            post_dce = post_dce.to(device)
            labels = labels_batch.to(device)
            loss = model(pre_dwi, pre_dce, post_dwi, post_dce, labels)
            accu_loss += loss # 累加损失
            data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))
    return accu_loss.item() / (step + 1)



 