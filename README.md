\# Federated Learning for Private Healthcare Diagnosis



\## 📌 Project Overview

This project focuses on building a \*\*privacy-preserving pneumonia detection system\*\* using \*\*Federated Learning\*\* and \*\*ResNet-18\*\* on chest X-ray images.



Instead of sending medical data to a central server, the model is trained in a \*\*distributed manner across multiple clients\*\*, ensuring patient data privacy while maintaining high accuracy.



\---



\## 🧠 Key Features



\- 🔒 Privacy-preserving learning using Federated Learning

\- 🫁 Pneumonia detection using ResNet-18 CNN model

\- 📊 Handles class imbalance in medical datasets

\- 🔄 Data augmentation to improve model generalization

\- 📍 Explainable AI using Grad-CAM visualizations

\- 📈 Performance evaluation across multiple clients



\---



\## ⚙️ Technologies Used



\- Python  

\- PyTorch  

\- Torchvision  

\- OpenCV  

\- Scikit-learn  

\- Federated Learning (custom / framework-based implementation)



\---



\## 📂 Dataset

Chest X-ray dataset containing:

\- Normal cases

\- Pneumonia cases



\---



\## 🚀 How It Works



1\. Dataset is split across multiple clients

2\. Each client trains the model locally

3\. Only model updates are shared (not data)

4\. Central server aggregates updates

5\. Final global model is evaluated



\---



\## 📊 Model Architecture

\- ResNet-18 backbone

\- Transfer learning applied

\- Fine-tuned for binary classification (Normal vs Pneumonia)



\---



\## 📌 Results

\- Improved privacy with no raw data sharing

\- Effective pneumonia classification performance

\- Visual explanations using Grad-CAM



\---

