# 🧠 CIFAR-10 Image Classification using CNN

This project demonstrates how to build and train a deep Convolutional Neural Network (CNN) using **TensorFlow/Keras** to classify images from the **CIFAR-10** dataset into 10 categories.

---

## 🎯 Objective

Build an accurate and generalizable CNN model to classify 32x32 color images into one of the following 10 classes:

- `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

---

## 📦 Dataset

- **Name**: CIFAR-10  
- **Source**: [Keras Datasets](https://www.cs.toronto.edu/~kriz/cifar.html)  
- **Size**: 60,000 images (50,000 training + 10,000 test)  
- **Image Shape**: (32, 32, 3)

---

## 🏗️ Model Architecture
The CNN model consists of 3 convolutional blocks followed by a fully connected (dense) layer.

```
Input: (32, 32, 3)
↓
Conv2D(64, 4x4, ReLU) + BatchNormalization

Conv2D(64, 4x4, ReLU) + BatchNormalization
MaxPooling2D(2x2)
Dropout(0.2)

Conv2D(128, 4x4, ReLU) + BatchNormalization
Conv2D(128, 4x4, ReLU) + BatchNormalization
MaxPooling2D(2x2)
Dropout(0.25)

Conv2D(128, 4x4, ReLU) + BatchNormalization
Conv2D(128, 4x4, ReLU) + BatchNormalization
MaxPooling2D(2x2)
Dropout(0.35)

Flatten
Dense(256, ReLU) + BatchNormalization
Dropout(0.5)
Dense(10, Softmax)

```

- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  
- **Evaluation Metric**: Accuracy  

---

## 📈 Training & Results

- **Epochs**: 25  
- **Best Validation Accuracy**: **86.34%** ✅  
- **Final Training Accuracy**: **96.54%**  
- **Validation Loss**: 0.53

### 🔍 Classification Report:

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Airplane | 0.85 | 0.90 | 0.87 |
| Automobile | 0.92 | 0.94 | 0.93 |
| Bird | 0.77 | 0.83 | 0.80 |
| Cat | 0.78 | 0.74 | 0.76 |
| Deer | 0.88 | 0.82 | 0.85 |
| Dog | 0.82 | 0.77 | 0.80 |
| Frog | 0.87 | 0.91 | 0.89 |
| Horse | 0.90 | 0.91 | 0.90 |
| Ship | 0.95 | 0.90 | 0.92 |
| Truck | 0.90 | 0.92 | 0.91 |

**Overall Accuracy on Test Set: 86%**

---

## 🛠️ Libraries Used

- Python 3
- TensorFlow / Keras
- Matplotlib, Seaborn
- scikit-learn
- NumPy / Pandas

---

## 🚀 Future Improvements

- Add **data augmentation** to improve generalization
- Implement **early stopping** and **learning rate scheduler**
- Try **transfer learning** (e.g., MobileNet, ResNet)
- Deploy the model using **Streamlit** or **Flask**

---

## 🙋 Author

- **Noran Ali**  
- [GitHub Profile](https://github.com/norannali)

---

## 💬 License

This project is open source and available under the [MIT License](LICENSE).


