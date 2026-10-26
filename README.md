# ❤️ Heart Disease Prediction using Machine Learning

> A comparative study of four supervised ML algorithms (KNN, Logistic Regression, Naive Bayes, SVM) that predict heart disease from patient clinical data. **SVM performed best with 83.7% accuracy and an F1-score of 0.86.**

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

Heart disease is one of the leading causes of death worldwide, and catching it early can save lives. Reading clinical data by hand is slow and error-prone, so this project builds a machine learning pipeline that sorts patients into **Healthy (0)** or **Heart Disease (1)** using non-invasive medical measurements such as age, cholesterol, blood pressure, and maximum heart rate.

Four classifiers are trained on the same preprocessed data and compared using accuracy, precision, recall, F1-score, and confusion matrices, to find the most reliable model for screening.

This was developed as my B.Tech CSE (AI) Project-1 at **Jamia Hamdard, New Delhi**, under the supervision of **Dr. Md. Tabrez Nafis**.

---

## 🎯 Key Highlights

- Compared **4 algorithms** on one dataset with an identical preprocessing pipeline
- Built with **scikit-learn `Pipeline` + `ColumnTransformer`** so test data never leaks into training
- Handled missing values, categorical encoding, and feature scaling
- Tuned **KNN** by testing K from 1 to 20 and plotting accuracy against K
- Evaluated with **confusion matrices** to see exactly how many sick patients each model caught or missed

---

## 📊 Dataset

- **Source:** UCI Heart Disease dataset (Kaggle version), combining records from **Cleveland, Hungary, Switzerland, and Long Beach**
- **Size:** 920 patient records, split **80% train / 20% test** (184 test samples)
- **Target:** original `num` (0 to 4) simplified into binary: `0` = Healthy, `1` = Disease present

| Feature | Meaning |
|---------|---------|
| `age` | Age in years |
| `sex` | Biological sex |
| `cp` | Chest pain type (typical angina, atypical angina, non-anginal, asymptomatic) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (True/False) |
| `restecg` | Resting ECG results |
| `thalch` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (True/False) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0 to 3) seen by fluoroscopy |
| `thal` | Thalassemia status (normal, fixed defect, reversible defect) |
| `id`, `dataset` | Identifiers, **dropped** before training |

---

## ⚙️ Methodology

```
Data Cleaning → Feature Transformation → Model Training → Evaluation
```

**Preprocessing**
1. **Drop** non-medical columns (`id`, `dataset`)
2. **Missing values:** median imputation for numeric columns, most-frequent (mode) imputation for categorical columns
3. **Encoding:** One-Hot Encoding for categorical features (e.g. chest pain type) so the model assumes no false order
4. **Scaling:** Standard Scaling so large-range features like `chol` (200+) don't overpower small ones like `oldpeak`
5. **Pipeline:** preprocessing and model are bundled together to prevent data leakage

---

## 🧠 Algorithms Used

| Algorithm | How it works | Notes from this project |
|-----------|--------------|-------------------------|
| **K-Nearest Neighbors** | Classifies a patient by majority vote of the *k* most similar patients (Euclidean distance) | Needs scaling; tested K = 1 to 20 |
| **Logistic Regression** | Weighted sum of features passed through a sigmoid to give a probability | Interpretable baseline |
| **Naive Bayes (Gaussian)** | Applies Bayes' Theorem assuming features are independent | Very fast; needs dense (non-sparse) input |
| **Support Vector Machine** | Finds the boundary with the widest margin between classes; RBF kernel handles non-linear patterns | Best performer, `C = 1.0` |

---

## 📈 Results

| Metric | Logistic Regression | KNN | Naive Bayes | **SVM** 🏆 |
|--------|:---:|:---:|:---:|:---:|
| Accuracy | 79.89% | 82.07% | 79.89% | **83.70%** |
| Precision (weighted) | 0.80 | 0.83 | 0.81 | **0.84** |
| Recall (weighted) | 0.80 | 0.82 | 0.80 | **0.84** |
| F1-Score (disease class) | 0.8246 | 0.8390 | 0.8213 | **0.8585** |

### Confusion matrices (test set of 184 patients)

| Model | Healthy correct | Healthy wrongly flagged | Sick patients missed | Sick patients caught |
|-------|:---:|:---:|:---:|:---:|
| Logistic Regression | 60 | 15 | 22 | 87 |
| KNN | 65 | 10 | 23 | 86 |
| Naive Bayes | 62 | 13 | 24 | 85 |
| **SVM** | 63 | 12 | **18** | **91** |

**Takeaways**
- **SVM** was the top performer on every metric and **missed the fewest sick patients (18)**, the most important error in medical screening.
- **KNN** came second and had the highest disease-class precision (0.90), meaning the fewest false alarms.
- **Logistic Regression and Naive Bayes** tied on accuracy, and Logistic Regression gives readable coefficients, which is valuable in medicine.
- For KNN, accuracy rises with K and peaks around **K = 17**, higher than the initial K = 5.

> 📷 **### Confusion Matrices

**KNN**

![KNN Confusion Matrix](knn_confusion_matrix.png)

**Naive Bayes**

![Naive Bayes Confusion Matrix](naive_bayes_confusion_matrix.png)

**SVM (best model)**

![SVM Confusion Matrix](svm_confusion_matrix.png)

**Logistic Regression**

![Logistic Regression Confusion Matrix](logistic_regression_confusion_matrix.png)

### KNN: Accuracy vs K

![KNN Accuracy vs K](knn_accuracy_vs_k.png)

>
> `![SVM Confusion Matrix](images/svm_confusion_matrix.png)`
> `![KNN Accuracy vs K](images/knn_accuracy_vs_k.png)`

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Pandas:** data loading, cleaning, one-hot encoding
- **Scikit-learn:** preprocessing, models, pipelines, metrics
- **Matplotlib:** confusion matrices and accuracy-vs-K plot
- **NumPy:** numerical computing
- **Jupyter Notebook:** experiments and visualization

---

## 📂 Project Structure

```
heart-disease-prediction/
├── data/
│   └── heart_disease_uci.csv
├── notebooks/
│   ├── knn.ipynb
│   ├── naive_bayes.ipynb
│   ├── svm.ipynb
│   └── logistic_regression.ipynb
├── images/
├── requirements.txt
└── README.md
```

*(Update file names to match your actual files.)*

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/maryam2620/<your-repo-name>.git
cd <your-repo-name>

# 2. (Optional) create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter and open a notebook
jupyter notebook
```

Dataset: download the **Heart Disease UCI** dataset from [Kaggle](https://www.kaggle.com) or the [UCI Repository](https://archive.ics.uci.edu) and place it in the `data/` folder.

---

## 🔮 Future Improvements

- Hyperparameter tuning with GridSearchCV and cross-validation
- Try ensemble models (Random Forest, XGBoost)
- Add model explainability (SHAP / feature importance) for clinical transparency
- Validate on larger, more diverse datasets
- Deploy as a simple web app (Flask) for risk prediction

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and is not a substitute for professional medical diagnosis.

---

## 📚 References

1. Kaggle, "Heart Disease UCI Dataset"
2. D. Dua and C. Graff, "UCI Machine Learning Repository," University of California, Irvine, 2019
3. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, 2011
4. C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, 1995
5. T. Cover and P. Hart, "Nearest neighbor pattern classification," *IEEE Trans. Information Theory*, 1967

---

## 👩‍💻 Author

**Maryam Ali**
B.Tech CSE (AI), Jamia Hamdard, New Delhi
GitHub: [@maryam2620](https://github.com/maryam2620)

⭐ If you found this useful, consider starring the repo!
