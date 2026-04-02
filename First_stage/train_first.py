import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from mydataset import MyDataSet, read_split_data_by_hospital, train_one_epoch, valid_one_epoch, get_train_transforms, get_test_transforms, save_fold_splits, first_plot_losses # 确保导入 MyDataSet
from model_first import DoubleTower_Delta as create_model
from sklearn.model_selection import StratifiedKFold # 导入用于内部验证集划分
import torch.nn as nn
import numpy as np
import random
import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./log/train.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main(args):
    # 为 reproducibility 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu")
    logger.info(f"Using device: {device}")

    # --- 数据加载和交叉验证部分 ---
    # 使用你的 read_split_data_by_hospital 函数获取用于交叉验证的数据池和外部测试集
    # train_hospitals 默认值就是 [3, 5]，所以这里可以不显式传入，但为了清晰，写上
    cv_pool_data, external_test_sets = read_split_data_by_hospital(
        csv_file_path=args.csv_path,
        image_root=args.data_path,
        train_hospitals=args.train_hospitals, # 默认是 [3, 5]
        random_seed=args.seed
    )

    logger.info(f"\n--- Cross-Validation Setup ---")
    logger.info(f"Total patients in CV pool (from hospitals {args.train_hospitals}): {len(cv_pool_data)}")
    if external_test_sets:
        logger.info(f"External test sets available for hospitals: {list(external_test_sets.keys())}")
        # 如果需要，可以在这里对外部测试集进行处理，例如创建 DataLoader

    # 从 cv_pool_data 中提取标签，用于 StratifiedKFold
    labels_for_skf = [item['label'] for item in cv_pool_data]
    
    # 初始化 StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    # 确保模型保存目录存在
    model_save_dir = args.model_save_dir
    os.makedirs(model_save_dir, exist_ok=True)
    cv_splits_dir = args.cv_split_dir
    os.makedirs(cv_splits_dir, exist_ok=True)
    
    # 存储每个折叠的训练和验证损失，以便后续分析
    all_fold_train_losses = []
    all_fold_val_losses = []

    # 遍历每个折叠
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(cv_pool_data)), labels_for_skf)):
        logger.info(f"\n==================== Fold {fold + 1}/{skf.n_splits} ====================")
        
        # 为每个折叠创建一个独立的 TensorBoard SummaryWriter
        fold_tb_writer = SummaryWriter(log_dir=os.path.join(f'runs/fold_{fold+1}'))

        # 保存当前折的划分
        split_df = save_fold_splits(fold, train_idx, val_idx, cv_pool_data, cv_splits_dir)

        # 根据索引从总数据池中获取当前折叠的训练和验证数据
        train_fold_data = [cv_pool_data[i] for i in train_idx]
        val_fold_data = [cv_pool_data[i] for i in val_idx]
        logger.info(f"Fold {fold + 1} Train samples: {len(train_fold_data)}, Validation samples: {len(val_fold_data)}")

        # 创建数据集和数据加载器
        img_size = args.img_size # 确保 args 中有 img_size
        train_preprocess = get_train_transforms(img_size)
        val_preprocess = get_test_transforms(img_size)
        
        train_dataset = MyDataSet(patient_data_list=train_fold_data, transform=train_preprocess)
        val_dataset = MyDataSet(patient_data_list=val_fold_data, transform=val_preprocess)

        batch_size = args.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        logger.info(f'Using {nw} dataloader workers for this fold')

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=nw,
                                                    collate_fn=train_dataset.collate_fn)
        
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=val_dataset.collate_fn)

        # 初始化模型、优化器等（每个折叠都重新初始化，以确保独立性）
        # model = create_model(in_channels=1, out_channels=1, contrast_margin=args.contrast_margin).to(device)
        model = create_model(in_channels=1, out_channels=1, num_down_blocks=3, contrast_margin=args.contrast_margin).to(device)
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

        best_fold_loss = float('inf')
        no_improve = 0
        patience = args.patience # 使用 args 中的 patience 参数

        # 为当前折叠的训练和验证损失创建新的列表，用于记录本折叠的所有 epoch 损失
        current_fold_train_losses_history = []
        current_fold_val_losses_history = []

        # 训练循环
        for epoch in range(args.epochs):
            
            train_loss = train_one_epoch(model=model, optimizer=optimizer,
                                         data_loader=train_loader,
                                         device=device,
                                         epoch=epoch)

            val_loss = valid_one_epoch(model=model, 
                                       data_loader=val_loader,
                                       device=device,
                                       epoch=epoch)

            logger.info(f"Fold {fold + 1} | Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # TensorBoard logging for the current fold
            fold_tb_writer.add_scalar("Train_Loss", train_loss, epoch)
            fold_tb_writer.add_scalar("Val_Loss", val_loss, epoch)
            fold_tb_writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

            # 记录当前 epoch 的损失，用于绘制本折叠的损失曲线
            current_fold_train_losses_history.append(train_loss)
            current_fold_val_losses_history.append(val_loss) # 记录当前的验证损失，而不是最佳验证损失


            if val_loss < best_fold_loss:
                best_fold_loss = val_loss
                no_improve = 0
                # 保存当前折叠的最佳模型
                model_save_path = os.path.join(model_save_dir, f"best_model_fold{fold+1}.pth")
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"Saved best model for Fold {fold+1} at epoch {epoch} with validation LOSS: {best_fold_loss:.3f}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f'Early stopping for Fold {fold+1} at epoch {epoch}')
                    break

            # --- 训练循环结束后，绘制并保存本折叠的损失曲线 ---
        first_plot_losses(current_fold_train_losses_history, current_fold_val_losses_history, fold, save_dir=args.cv_split_dir)
        logger.info(f"Saved loss curve for Fold {fold+1} to {os.path.join(args.cv_split_dir, f'first_fold_{fold + 1}_loss_curve.png')}")

        # 记录当前折叠的最终训练损失和最佳验证损失，用于计算所有折叠的平均值
        all_fold_train_losses.append(current_fold_train_losses_history[-1]) # 记录最后一轮的训练损失
        all_fold_val_losses.append(best_fold_loss) # 记录该折叠的最佳验证损失

        fold_tb_writer.close() # 关闭当前折叠的 TensorBoard writer

    logger.info(f"\n--- Cross-Validation Training Finished ---")
    logger.info(f"Average training loss across {skf.n_splits} folds: {np.mean(all_fold_train_losses):.4f}")
    logger.info(f"Average best validation loss across {skf.n_splits} folds: {np.mean(all_fold_val_losses):.4f}")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gpu_id', type=int, default=1, help='使用的GPU编号，-1表示使用CPU')
    parser.add_argument('--img_size', type=int, default=224, help='图像尺寸') # 新增图像尺寸参数
    parser.add_argument('--patience', type=int, default=8, help='早停的耐心值') # 新增 patience 参数

    # 数据集相关参数
    parser.add_argument('--csv_path', type=str,
                        default="./dataset_csv/new_1292.csv",
                        help='Path to the CSV file containing patient info and labels')
    parser.add_argument('--data_path', type=str,
                        default="/data/data/FromHospital/YiZhong-HE-Neoadjuvant-Breast/Clean_Data/Again_croppng",
                        help='Root directory for processed MRI images')
    
    # 划分参数
    parser.add_argument('--train_hospitals', type=int, nargs='+', default=[3, 5],
                        help='List of hospital IDs to use for training (e.g., --train_hospitals 3 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--contrast_margin', type=float, default=2.0, help='Random seed for reproducibility')

    # 调整权重保存目录，以便保存每个折叠的模型
    parser.add_argument('--model_save_dir', type=str, default='./weights/first_weight', help='Directory to save model weights for each fold')
    parser.add_argument('--cv_split_dir', type=str, default='./cv_split', help='模型保存目录')
    parser.add_argument('--loss_path', type=str, default="./cv_split/training_losses.npy",
                        help='Path to save training loss numpy array (this is not currently used to save epoch loss in main)')

    opt = parser.parse_args()

    main(opt)
