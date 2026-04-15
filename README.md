# TSGMP
Title: A Two-Stage Multimodal Prediction Method for Neoadjuvant Therapy Response in Breast Cancer Based on Dynamic Imaging Features

# Abstract: 
Pathologic complete response (pCR) is a critical indicator for evaluating neoadjuvant therapy (NAT) efficacy in breast cancer, yet existing prediction models often rely on post-treatment data, limiting their utility for pre-treatment decision-making. In this study, we introduce the Two-Stage Multimodal Prediction (TSGMP) model, a novel framework designed to achieve precise pCR prediction using solely pre-treatment data. Guided by a 'Training-on-Dynamic and Prediction-on-Static' paradigm, we pre-train a multi-task encoder on longitudinal multi-parametric MRI to extract dynamic tumor response signatures. These signatures are then synergistically fused with radiomics and clinicopathological features via a cross-modal attention module to achieve accurate pCR prediction. We developed and validated TSGMP using a large-scale retrospective cohort from five independent medical centers. The proposed model achieved an AUC of 0.851 in the primary cohort and demonstrated robust generalizability across three external validation cohorts (AUCs of 0.804, 0.813, and 0.787). Furthermore, TSGMP significantly outperformed state-of-the-art methods and single-modality approaches; subgroup analyses further confirmed its consistent stability across diverse molecular subtypes. These findings underscore the potential of TSGMP as a powerful, interpretable tool for personalized NAT decision-making, effectively bridging the gap between dynamic training data and static clinical application.

# 📊 Data
Due to privacy and ethical regulations, the original patient data from the multi-center study cannot be shared publicly. The Data_sample/directory provides example data formats and scripts to help users structure their own data for the pipeline.

A general data preparation pipeline includes:
- Medical Images:​ Pre-treatment multi-parametric MRI (e.g., DWI, DCE sequences). For the first stage, longitudinal scans are needed.
- Radiomics Features:​ Extracted from the Region of Interest (ROI) on pre-treatment images using tools like pyradiomics.
- Clinical Data:​ Tabular data (e.g., age, subtype, tumor grade).

To reproduce the key results reported in the paper: Prepare your dataset according to the structure outlined in Data_sample/.

