# Task 7: Support Vector Machines (SVM) – Breast Cancer Classification

## Objective
To apply Support Vector Machines (SVM) for binary classification on the Breast Cancer dataset.  
Both linear and non-linear (RBF) kernels are tested, along with hyperparameter tuning and cross-validation.  

---

## Tools Used
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## Dataset
Dataset: **Breast Cancer Dataset (Wisconsin)**  

- **Target Variable (diagnosis):**
  - `M` → Malignant (Cancer Present, encoded as `1`)  
  - `B` → Benign (No Cancer, encoded as `0`)  

- **Features:**  
  - `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`,  
  - `concavity_mean`, `concave points_mean`, `symmetry_mean`, `fractal_dimension_mean`,  
  - … and their respective SE (standard error) and worst-case values.  

---

## Steps Performed

### 1. Data Preprocessing
- Dropped the `id` column (not useful).  
- Encoded target (`M=1`, `B=0`).  
- Standardized features using `StandardScaler`.  
- Split dataset into 80% training and 20% testing sets.  

### 2. Linear SVM
- Trained `SVC(kernel='linear')`.  
- Evaluated accuracy, confusion matrix, and classification report.  

### 3. Non-linear SVM (RBF Kernel)
- Trained `SVC(kernel='rbf')`.  
- Compared performance with linear kernel.  

### 4. Hyperparameter Tuning
- Used `GridSearchCV` to tune `C` and `gamma`.  
- Found best parameters for RBF kernel.  

### 5. Cross-Validation
- Performed 5-fold cross-validation.  
- Reported mean CV accuracy for robustness.  

### 6. Visualization
- For demonstration, used only **2 features** (`radius_mean`, `texture_mean`) to plot decision boundaries.  
- Plotted SVM classification regions using Matplotlib.  

## Prepared By
Tanuja Deshmukh  
AI & ML Internship – Task 7
