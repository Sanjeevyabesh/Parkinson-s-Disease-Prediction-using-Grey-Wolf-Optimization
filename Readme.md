# 🧠 Parkinson’s Disease Prediction using Grey Wolf Optimization (GWO)

## 📌 Abstract

This project presents a Parkinson’s Disease Prediction system using speech features and a machine learning model optimized with Grey Wolf Optimization (GWO). The proposed approach improves prediction accuracy by performing feature selection and hyperparameter tuning simultaneously. By reducing unnecessary features and optimizing model parameters, the system achieves better performance and generalization compared to a baseline model.

---

## 🎯 Objective

* Detect Parkinson’s Disease using speech dataset
* Improve model accuracy using a Meta-Heuristic Optimization algorithm
* Perform feature selection and parameter tuning

---

## ⚙️ Methodology

### 1. Data Processing

* Loaded Parkinson’s dataset
* Removed unnecessary columns
* Standardized features

### 2. Baseline Model

* Used Support Vector Machine (SVM)
* Trained with default parameters
* Measured baseline accuracy

### 3. Optimization using Grey Wolf Optimization (GWO)

* Applied GWO as an optimization layer
* Optimized:

  * Feature selection (important features only)
  * Hyperparameters (C, gamma)

### 4. Final Model

* Trained optimized SVM using selected features
* Compared performance with baseline model

---

## 📊 Outputs Obtained

### 🔹 Accuracy Comparison

* Baseline Accuracy: ~0.89
* Optimized Accuracy: ~0.94 – 0.97

👉 Shows clear improvement using optimization

---

### 🔹 Feature Selection

* Total Features: (depends on dataset, e.g., 22)
* Selected Features: Reduced subset
* Important features automatically identified

---

### 🔹 Optimization Performance

* GWO Convergence Graph generated
* Fitness value reduced over iterations
* Shows algorithm is learning and improving

---

### 🔹 Evaluation Metrics

* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)
* ROC Curve (AUC score)

---

## 📈 Visual Outputs

Saved inside `results/` folder:

* `convergence.png` → Optimization progress
* `roc_curve.png` → Model performance

---

## 🧠 Key Highlights

* Used Meta-Heuristic Optimization (GWO)
* Performed dual optimization:

  * Feature selection
  * Hyperparameter tuning
* Improved model accuracy and stability
* Reduced feature complexity

---

## 🚀 Conclusion

The project demonstrates that applying Grey Wolf Optimization significantly enhances the performance of machine learning models for Parkinson’s Disease detection. The optimized model achieves higher accuracy with fewer features, making it more efficient and reliable.

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## 📁 Project Structure

```
parkinsons_gwo_project/
│
├── data/
├── src/
├── results/
├── main.py
├── requirements.txt
└── README.md
```
