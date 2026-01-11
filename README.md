# ML_practice_projects

👋 Hi, I’m **Nikhil**.  
This repository is a collection of my **machine learning practice projects**, built to deeply understand ML concepts by implementing them **from scratch**, not just using libraries.

---

## 📌 Project: Linear Regression from Scratch (Property Price Prediction)

### 🔍 Overview
In this project, I implemented **Linear Regression using Gradient Descent** from scratch to predict **property prices based on property size**.

The goal was not just to get predictions, but to **understand how linear regression actually works internally**, including:
- cost minimization
- gradient descent behavior
- feature scaling
- learning rate tuning
- prediction on new data

---

## 🧠 What I Learned / Implemented

### ✅ Core ML Concepts
- Linear Regression model:  
  \[
  y = wx + b
  \]
- Mean Squared Error cost function
- Gradient computation for `w` and `b`
- Gradient Descent optimization

### ✅ Practical ML Skills
- Feature scaling (basic scaling)
- Choosing and tuning learning rate
- Debugging exploding gradients and NaNs
- Tracking cost over iterations
- Making predictions on unseen inputs
- Converting predictions back to real-world units

---

## 📊 Visualizations Included
- Raw data plot (property size vs price)
- Regression line vs actual data
- Cost vs iterations (Gradient Descent progress)
- Actual vs Predicted comparison
- 3D visualization of cost surface with gradient descent path

These plots helped in **visually verifying** that the model is learning correctly.

---

## 🏠 Dataset
- **Input (`x`)**: Property size (sq ft)
- **Output (`y`)**: Property cost (in lakhs)
- Data was scaled before training to ensure stable gradient descent.

---

## 🛠️ Tech Stack
- Python
- NumPy
- Matplotlib

(No ML libraries like scikit-learn were used for training — everything is implemented manually.)

---

## 🚀 Why This Project Matters
This project focuses on **fundamentals**, not shortcuts.  
By building everything from scratch, I gained:
- strong intuition for optimization
- understanding of why scaling is necessary
- confidence to debug ML models

This serves as a foundation for:
- multivariable linear regression
- logistic regression
- neural networks

---

## 📈 Future Work
- Multivariable Linear Regression
- Vectorized Gradient Descent
- Train/Test split and evaluation
- Comparison with `sklearn`
- Regularization (Ridge & Lasso)

---

## ✍️ Author
**Nikhil**  
Computer Science student  
Learning ML by building from first principles 🚀
