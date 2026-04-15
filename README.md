# TSGMP
**Title:** A Two-Stage Multimodal Prediction Method for Neoadjuvant Therapy Response in Breast Cancer Based on Dynamic Imaging Features

# Abstract: 
Pathologic complete response (pCR) is a critical indicator for evaluating neoadjuvant therapy (NAT) efficacy in breast cancer, yet existing prediction models often rely on post-treatment data, limiting their utility for pre-treatment decision-making. In this study, we introduce the Two-Stage Multimodal Prediction (TSGMP) model, a novel framework designed to achieve precise pCR prediction using solely pre-treatment data. Guided by a 'Training-on-Dynamic and Prediction-on-Static' paradigm, we pre-train a multi-task encoder on longitudinal multi-parametric MRI to extract dynamic tumor response signatures. These signatures are then synergistically fused with radiomics and clinicopathological features via a cross-modal attention module to achieve accurate pCR prediction. We developed and validated TSGMP using a large-scale retrospective cohort from five independent medical centers. The proposed model achieved an AUC of 0.851 in the primary cohort and demonstrated robust generalizability across three external validation cohorts (AUCs of 0.804, 0.813, and 0.787). Furthermore, TSGMP significantly outperformed state-of-the-art methods and single-modality approaches; subgroup analyses further confirmed its consistent stability across diverse molecular subtypes. These findings underscore the potential of TSGMP as a powerful, interpretable tool for personalized NAT decision-making, effectively bridging the gap between dynamic training data and static clinical application.

# 🚀 Usage
The workflow follows the two-stage architecture.

1. Data Preprocessing and Feature Extraction

This step prepares the static pre-treatment data for the second stage.

**- /Data_process/crop.py:** Crops the tumor Region of Interest (ROI) from medical images.

**- /Data_process/radiomics_feature_extractor.py:** Extracts hand-crafted radiomics features.

2. First Stage - Dynamic Encoder Pre-training

This stage trains an encoder on longitudinal​ MRI data to learn generalizable dynamic features.

**- /First_stage/train_first.py:** Training script for the first-stage multi-task encoder. It processes longitudinal data to capture tumor evolution patterns.

3. Second Stage - Multimodal Fusion and Prediction

This stage uses pre-processed data, clinical data, and extracted radiomics features for final pCR prediction.

**- /Second_stage/main.ipynb:** Jupyter Notebook containing the complete pipeline for the second stage. It includes:

    - Loading the pre-trained encoder from First Stage

    - Fusing dynamic features with radiomics and clinical features

    - Training the cross-modal attention model

    - Evaluating performance and generating predictions

# 📊 Data
Due to privacy and ethical regulations, the original patient data from the multi-center study cannot be shared publicly. However, to facilitate research, we are providing processed, de-identified data.​ This dataset includes pre- and post-treatment cropped ROI images, clinical data, and extracted radiomics features from 1292 patients across five medical centers. You can download it from the following link: [Dataset Release (GitHub)](https://github.com/sufaer/TSGMP/releases/tag/dataset).

**Note:**​ The provided data is in a processed format for model training and validation. The Data_sample/directory provides example scripts to illustrate the data structure and facilitate the use of your own data.

A general data preparation pipeline includes:

**- Medical Images:**​ Pre-treatment multi-parametric MRI (e.g., DWI, DCE sequences). For the first stage, longitudinal scans are needed.

**- Radiomics Features:**​ Extracted from the Region of Interest (ROI) on pre-treatment images using tools like pyradiomics.

**- Clinical Data:**​ Tabular data (e.g., age, subtype, tumor grade).

To reproduce the key results reported in the paper: Prepare your dataset according to the structure outlined in Data_sample/.
