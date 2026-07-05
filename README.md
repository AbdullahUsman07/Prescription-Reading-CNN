# Handwritten Doctor Prescription Reader using Custom CNN

An end-to-end Deep Learning project designed to recognize and classify handwritten doctor prescriptions into 78 distinct medicine classes. 

## 📊 Dataset
The dataset utilized for this project contains approximately 1,000 images of handwritten medical prescriptions. Due to file size constraints, the raw data is not hosted in this repository. 
* You can access and download the original dataset here: [https://www.kaggle.com/datasets/nadaarfaoui/ocr-processed-handwritten-prescriptions]

## 🧠 Developed Solution & Model Architecture

### Solution 1: Custom Convolutional Neural Network (CNN)
A custome lightweight architecutre built from scratch using TensorFlow/Keras. It features a hierarchical block design:
* **Block 1 (Low-level):** 32 filters (3x3), Batch Normalization, MaxPooling, Dropout (0.25) to extract primitive strokes.
* **Block 2 (Medium-level):** 64 filters (3x3), Batch Normalization, MaxPooling, Dropout (0.25) to catch intersecting shapes and loops.
* **Block 3 (High-level):** 128 filters (3x3), Batch Normalization, MaxPooling, Dropout (0.25) for overall word/character structures.
* **Classifier Brain:** Flatten layer, Dense (256 nodes with 0.50 Dropout to combat heavy handwriting overfitting), and a 78-node Softmax output layer.

### Solution 2: ResNet-50: Transfer Learning Backbone
To scale performance and run a comparative analysis, a second solution was developed using state-of-the-art **Resnet-50 (Residual Network)** architecture pre-trained on the ImageNet dataset.
* **Feature Extraction Base:** The core convolutional layers of ResNet-50 are utilized as a robust feature extractor. Because the early layers of ResNet-50 are pre-trained experts at identifying edges, slants, and visual lines, its entire **base parameter weight set was frozen** (`trainable = False`). This locked its general visual motor skills, preventing our small custom dataset from over-fitting or erasing its pre-learned features.
* **Custom Adaptation:** The input channel size was adjusted to accept our grayscale resolution, and a custom classification head was appended to the base: Global Average Pooling $\rightarrow$ Dense (256 nodes) $\rightarrow$ Dropout (0.50) $\rightarrow$ 78-node Softmax Output.

---

## ⚙️ Data Pipeline & Training Optimization
Both models utilize an automated, dynamic processing pipeline to tackle low handwriting style diversity:
* **Real-time Augmentation:** Integrates on-the-fly mutations including random tilts (`rotation_range=12`), grid adjustments (`width/height shifts=0.1`), and cursive slanting (`shear_range=0.15`).
* **Adaptive Optimization:** Regulated via Keras Callbacks including `EarlyStopping` (patience=6 to prevent late-stage memorization) and `ReduceLROnPlateau` (which dynamically slices the learning rate by half whenever validation loss stalls, smoothing out volatile overshooting).

---

## 📊 Comparative Performance Results

| Model Architecture | Optimization Curve | Test Accuracy | Test Loss Status |
| :--- | :--- | :--- | :--- |
| **Custom CNN from Scratch** | Highly volatile early on; successfully stabilized via adaptive LR cooling at Epoch 12. | **74.23%** | **1.2629** (Smart, low-confidence errors) |
| **ResNet-50 (Transfer Learning)** | Exceptionally stable learning trajectory, lightning-fast training loops due to frozen parameters. | **52.44%** | **2.1677** |

---


## 🚀 How to Run on Google Colab
1. Upload the notebook `Doctor_Prescription_CNN.ipynb` to Google Colab.
2. Download the dataset zip from Kaggle and upload it directly to Colab's high-speed local VM storage (`/content/`).
3. Unzip the dataset locally using `!unzip` for maximum training speed.
4. Run the notebook cells sequentially.