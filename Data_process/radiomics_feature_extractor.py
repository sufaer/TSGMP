import radiomics
from radiomics import featureextractor
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import glob
import six
import logging
from tqdm import tqdm

# 日志配置
logging.basicConfig(
    filename='radiomics_feature.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_nii_file(seq_dir):
    """查找目录中的NIfTI文件，优先返回.nii文件"""
    nii_files = glob.glob(os.path.join(seq_dir, "*.nii"))
    nii_gz_files = glob.glob(os.path.join(seq_dir, "*.nii.gz"))
    all_nii_files = nii_files + nii_gz_files
    if not all_nii_files:
        raise FileNotFoundError(f"在目录 {seq_dir} 中未找到NIfTI文件")
    # 优先返回.nii文件
    for file in all_nii_files:
        if file.endswith('.nii'):
            return file
    return all_nii_files[0]

def feature_extractor(csv_path, image_dir1, image_dir2, output_csv):
    """
    放射组学特征提取主函数
    :param csv_path: 包含患者ID和标签路径的CSV文件
    :param image_dir1: 治疗前影像目录
    :param image_dir2: 治疗后影像目录
    :param output_csv: 特征保存路径
    """
    # 1. 加载患者数据
    label_df = pd.read_csv(csv_path, encoding='gbk')
    label_df['patient_ID'] = label_df['patient_ID'].astype(str).str.strip()
    label_df = label_df.dropna(subset=['patient_ID'])
    # label_df = label_df[label_df['hospital'].isin([3, 5])]
    # label_df = label_df[label_df['hospital'].isin([1,2,4])]
    
    # 2. 初始化特征提取器
    params_path = 'Params.yaml'
    extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
    
    # 3. 准备结果存储
    all_features = []
    feature_names = None
    
    # 4. 遍历所有患者
    for _, row in tqdm(label_df.iterrows(), total=len(label_df), desc="Processing patients"):
        pid = str(row['patient_ID']).strip()
        features = {'patient_ID': pid}

        # 定义影像-标签映射（治疗前/后，DCE/DWI）
        image_label_mapping = [
            # (影像路径, 标签路径, 特征前缀)
            (os.path.join(image_dir1, f"{pid}_c1.nii.gz"),   row['c1'],   "preDCE"),
            (os.path.join(image_dir2, f"{pid}_c2_reg.nii.gz"), row['c1'], "postDCE"),
            (os.path.join(image_dir1, f"{pid}_dwi1.nii.gz"), row['dwi1'], "preDWI"),
            (os.path.join(image_dir2, f"{pid}_dwi2_reg.nii.gz"), row['dwi1'], "postDWI")
            # (os.path.join(image_dir1, f"{pid}_c1.nii.gz"), find_nii_file(row['c1']),   "preDCE"),
            # (os.path.join(image_dir1, f"{pid}_c2.nii.gz"), os.path.join(image_dir2, f"{pid}_c2_mask.nii.gz"), "postDCE"),
            # (os.path.join(image_dir1, f"{pid}_adc1.nii.gz"), find_nii_file(row['dwi1']), "preADC"),
            # (os.path.join(image_dir1, f"{pid}_adc2.nii.gz"), os.path.join(image_dir2, f"{pid}_dwi2_mask.nii.gz"), "postADC")
        ]

        for img_path, label_dir, prefix in image_label_mapping:
            try:
                if not os.path.exists(img_path):
                    logger.warning(f"影像文件不存在: {img_path}")
                    continue
                
                # label_path = find_nii_file(label_dir)
                label_path = label_dir
                radiomics_features = extractor.execute(img_path, label_path)
                
                # 首次运行时初始化特征名
                if feature_names is None:
                    feature_names = [k for k in radiomics_features.keys() 
                                   if not k.startswith('diagnostics')]
                
                # 添加前缀存储特征
                for feat_name in feature_names:
                    features[f"{prefix}_{feat_name}"] = radiomics_features.get(feat_name, np.nan)
                
            except Exception as e:
                logger.error(f"患者 {pid} {prefix} 处理失败: {str(e)}")
                # 标记失败的特征为NA
                for feat_name in feature_names or []:
                    features[f"{prefix}_{feat_name}"] = np.nan
        
        all_features.append(features)

    
    # 5. 保存结果
    if all_features:
        result_df = pd.DataFrame(all_features)
        result_df.to_csv(output_csv, index=False)
        logger.info(f"成功提取 {len(result_df)} 个患者的特征，保存至 {output_csv}")
    else:
        logger.warning("未提取到任何特征数据")

# 使用示例
if __name__ == "__main__":
    # 路径配置
    image_dir1 = '/data/data/FromHospital/YiZhong-HE-Neoadjuvant-Breast/Clean_Data/DicomtoNii_all'
    image_dir2 = '/data/data/FromHospital/YiZhong-HE-Neoadjuvant-Breast/Clean_Data/registration_post'
    csv_path = 'new_1304.csv'
    output_csv = 'radiomics_features.csv'
    
    # 执行特征提取
    feature_extractor(csv_path, image_dir1, image_dir2, output_csv)
    