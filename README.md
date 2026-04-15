# CO₂ Emission Prediction (Practice Project)

This is a **machine learning practice project** where I explore how to predict **CO₂ emissions** using structured data.

The main goal of this project is to practice:
- data preprocessing
- building ML pipelines
- training regression models
- understanding model performance

---

## 📌 Overview

In this project, I use a dataset containing features related to fuel consumption and vehicle characteristics to predict:

👉 **CO₂ Emissions**

This is a supervised regression problem.

---

## 📊 Dataset

The dataset includes features such as:

- Engine size
- Cylinders
- Fuel consumption
- Fuel type
- CO₂ emissions (target)

Target variable:
- `CO2 Emissions`

---

## ⚙️ What I Practised

### 1. Data Handling
- Load dataset with `pandas`
- Inspect structure and values

### 2. Preprocessing
- Handle numerical and categorical data
- Use:
  - `SimpleImputer`
  - `StandardScaler`
  - `OneHotEncoder`
- Combine with `ColumnTransformer`

### 3. Model Building
- Train regression models (mainly from `sklearn`)
- Use pipelines to simplify workflow

### 4. Model Evaluation
- Train/test split
- Evaluate using:
  - R² score
  - Mean Squared Error

### 5. Experimentation
- Try different models and parameters
- Compare performance

---

## 📁 Project Structure

```bash
Co2-project/
├── *.ipynb        # Jupyter notebooks (experiments)
├── *.py           # Python scripts for training/testing
├── dataset file   # CO₂ dataset
└── README.md
