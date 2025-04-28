# ML Visualization Playground

![ML Visualization Badge](https://img.shields.io/badge/ML-Visualization-blue)
![Python Version](https://img.shields.io/badge/python-3.8%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.44.0%2B-FF4B4B)

An interactive web application for exploring and visualizing fundamental machine learning and statistical concepts. This playground makes complex statistical and machine learning concepts intuitive through dynamic, interactive visualizations.

## ✨ Features

- **Normal Distribution Explorer**: Interact with parameters to understand normal distribution properties
- **Central Limit Theorem Visualization**: See sampling distributions approach normality regardless of source distribution
- **Linear Regression Explorer**: Experiment with model fitting and visualize the loss landscape in 3D
- **Polynomial Regression with SGD**: Watch stochastic gradient descent optimize polynomial models in real-time

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🚀 Installation & Setup

### Clone the repository

```bash
git clone https://github.com/88VwV88/ml-fest-playground.git
cd ml-fest-playground
```

### Set up a virtual environment (recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run main.py
```

The application should open automatically in your default web browser. If not, navigate to the URL displayed in the terminal (typically http://localhost:8501).

## 📊 Usage Guide

### Normal Distribution

- Adjust mean and standard deviation using sliders
- See how parameters affect the shape of the distribution
- View probability ranges within standard deviations

### Central Limit Theorem

- Choose from different underlying distributions (uniform, exponential, bimodal, etc.)
- Change sample size and number of samples
- Observe how sample means approach a normal distribution

### Linear Regression

- Generate data with controllable noise levels
- Adjust model weight and bias parameters
- Explore the loss surface in interactive 3D plots
- Understand how parameter changes affect model fit

### Polynomial Regression with SGD

- Select from various underlying functions (quadratic, cubic, sinusoidal, exponential)
- Configure polynomial degree and learning parameters
- Watch the model learn in real-time with stochastic gradient descent
- View the convergence process and loss curve

## 🔧 Dependencies

This project relies on the following libraries:

- streamlit >= 1.44.1
- numpy >= 2.2.5
- pandas >= 2.2.3
- matplotlib >= 3.10.1
- plotly >= 6.0.1
- seaborn >= 0.13.2
- scikit-learn >= 1.6.1
