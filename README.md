# Handwritten Doctor Prescription Reader using Custom CNN

An end-to-end Deep Learning project designed to recognize and classify handwritten doctor prescriptions into 78 distinct medicine classes. 

## 📊 Dataset
The dataset utilized for this project contains approximately 1,000 images of handwritten medical prescriptions. Due to file size constraints, the raw data is not hosted in this repository. 
* You can access and download the original dataset here: [Link to Kaggle Dataset]

## 🧠 Model Architecture
The model is a custom Convolutional Neural Network (CNN) built from scratch using TensorFlow/Keras. It features a hierarchical block design:
* **Block 1 (Low-level):** 32 filters (3x3), Batch Normalization, MaxPooling, Dropout (0.25) to extract primitive strokes.
* **Block 2 (Medium-level):** 64 filters (3x3), Batch Normalization, MaxPooling, Dropout (0.25) to catch intersecting shapes and loops.
* **Block 3 (High-level):** 128 filters (3x3), Batch Normalization, MaxPooling, Dropout (0.25) for overall word/character structures.
* **Classifier Brain:** Flatten layer, Dense (256 nodes with 0.50 Dropout to combat heavy handwriting overfitting), and a 78-node Softmax output layer.

## 🚀 How to Run on Google Colab
1. Upload the notebook `Doctor_Prescription_CNN.ipynb` to Google Colab.
2. Download the dataset zip from Kaggle and upload it directly to Colab's high-speed local VM storage (`/content/`).
3. Unzip the dataset locally using `!unzip` for maximum training speed.
4. Run the notebook cells sequentially.