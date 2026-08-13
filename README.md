# ⚛️ Quantum Machine Learning on the Iris Dataset

> **A beginner-to-intermediate Quantum Machine Learning project that compares classical Support Vector Classification with Variational Quantum Classifiers (VQC) using the classic Iris dataset.**

This repository presents a hands-on introduction to **Quantum Machine Learning (QML)** through one of the most well-known datasets in machine learning: the **Iris flower dataset**.

The project is intentionally built around a familiar, relatively small dataset so that the focus remains on understanding the **quantum machine learning workflow**, rather than struggling with a complicated real-world dataset.

The central question explored throughout the notebook is:

> **Can a Quantum Machine Learning model learn a simple classical classification problem, and how does its performance compare with a conventional machine learning model?**

To answer this, the project builds and evaluates:

* A classical **Support Vector Classifier (SVC)**
* A **Variational Quantum Classifier (VQC)** using all four features
* A VQC using PCA-reduced two-dimensional data
* A VQC using the `RealAmplitudes` ansatz
* A VQC using the `EfficientSU2` ansatz

The project therefore provides a practical bridge between **classical Machine Learning and Quantum Machine Learning**.

---

# 📌 Project Summary

This project takes a deliberately familiar machine learning problem—the **Iris classification task**—and uses it as a laboratory for understanding Quantum Machine Learning.

The progression is:

```text
Iris Dataset
     ↓
EDA
     ↓
Feature Scaling
     ↓
Classical SVC
     ↓
ZZ Feature Map
     ↓
RealAmplitudes Ansatz
     ↓
VQC
     ↓
PCA
     ↓
Reduced-Feature VQC
     ↓
EfficientSU2
     ↓
Classical vs Quantum Comparison
```

The experiments demonstrate that the quantum models can successfully learn the classification task, while also showing that **model architecture, feature representation, and dimensionality can have a major impact on performance**.

Most importantly, the project provides a practical first step from conventional Machine Learning into **Quantum Machine Learning**, without requiring a complicated or domain-specific dataset.

---

# 🌟 Why the Iris Dataset?

The Iris dataset is one of the most widely used introductory datasets in Machine Learning.

It contains measurements of iris flowers belonging to three species:

* *Iris setosa*
* *Iris versicolor*
* *Iris virginica*

Each observation contains four numerical features:

| Feature      | Description         |
| ------------ | ------------------- |
| Sepal Length | Length of the sepal |
| Sepal Width  | Width of the sepal  |
| Petal Length | Length of the petal |
| Petal Width  | Width of the petal  |

The dataset contains:

* **150 samples**
* **4 numerical features**
* **3 target classes**
* **50 samples per class**

Because the dataset is small and well understood, it is particularly suitable for demonstrating the mechanics of a QML model.

> **Note:** The dataset is intentionally "commoditized" / highly familiar. The purpose of this project is not to solve a novel business problem, but to provide a clean and understandable environment for learning Quantum Machine Learning.

---

# 🎯 Project Objectives

The project aims to:

* Introduce the basic workflow of Quantum Machine Learning.
* Explore a classical classification problem.
* Establish a classical SVC benchmark.
* Build a Variational Quantum Classifier.
* Understand quantum feature maps.
* Understand parameterized quantum circuits.
* Experiment with different quantum ansätze.
* Investigate the effect of dimensionality reduction.
* Compare classical and quantum model performance.
* Demonstrate the practical limitations of small QML models.

---

# 🔄 Overall Workflow

```text
                Iris Dataset
                     │
                     ▼
             Data Exploration
                     │
                     ▼
              Feature Scaling
                     │
                     ▼
             Train/Test Split
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
 Classical ML              Quantum ML
          │                     │
          ▼                     ▼
         SVC              ZZ Feature Map
                                │
                                ▼
                             Ansatz
                                │
                                ▼
                              VQC
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
          4-Feature Model              PCA 2-Feature Model
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                         Model Comparison
```

---

# 📂 Dataset

The dataset is loaded directly from Scikit-learn:

```python
from sklearn.datasets import load_iris

iris_data = load_iris()
```

The notebook explores the built-in dataset description before beginning the modeling process.

The features are extracted using:

```python
features = iris_data.data
labels = iris_data.target
```

---

# 🔍 Exploratory Data Analysis

The project begins with basic EDA to understand the structure of the dataset.

A Pandas DataFrame is created containing the four flower measurements and the corresponding class label.

A **Seaborn pairplot** is then used to visualize relationships between the features.

One of the important observations is that:

* Class `0` is relatively easy to separate from the other two classes.
* Classes `1` and `2` show greater overlap.

This makes Iris a useful introductory classification problem: it is simple enough for beginners but still contains a meaningful classification challenge.

---

# ⚙️ Data Preprocessing

Before training the models, the features are normalized using:

```python
from sklearn.preprocessing import MinMaxScaler
```

The scaling process maps the input features into a common numerical range.

This is particularly important for the quantum model because the classical feature values are eventually used as parameters in quantum circuits.

---

# ✂️ Train-Test Split

The dataset is divided into training and testing subsets.

The same general dataset split is used when comparing the classical and quantum models so that their performance can be evaluated on comparable data.

The project therefore follows the standard:

```text
Training Data
     ↓
Model Training
     ↓
Unseen Test Data
     ↓
Performance Evaluation
```

workflow.

---

# 🤖 Classical Machine Learning Benchmark

Before introducing quantum machine learning, the project establishes a classical baseline.

This is important because a quantum model should not be evaluated in isolation.

The classical model used is:

> **Support Vector Classifier (SVC)**

from Scikit-learn.

```python
from sklearn.svm import SVC

svc = SVC()
svc.fit(train_features, train_labels)
```

---

# 📊 Classical SVC Performance

Using all four Iris features, the classical SVC achieves:

| Model            | Training Score | Test Score |
| ---------------- | -------------: | ---------: |
| SVC — 4 Features |       **0.99** |   **0.97** |

This provides a strong baseline for evaluating the quantum models.

The result also demonstrates that the Iris classification problem is relatively easy for a conventional machine learning algorithm.

---

# ⚛️ Quantum Machine Learning

The second part of the project introduces a **Variational Quantum Classifier (VQC)**.

The basic architecture is:

```text
Classical Features
       │
       ▼
  Quantum Feature Map
       │
       ▼
 Trainable Ansatz
       │
       ▼
 Quantum Measurement
       │
       ▼
     VQC Output
       │
       ▼
 Classification
```

The VQC is implemented using **Qiskit Machine Learning**.

---

# 🗺️ Quantum Feature Map

The project uses the:

> **ZZ Feature Map**

from Qiskit's circuit library.

```python
from qiskit.circuit.library import ZZFeatureMap

feature_map = ZZFeatureMap(
    feature_dimension=num_features,
    reps=1
)
```

The feature map transforms the classical Iris features into a quantum representation.

Conceptually:

```text
Iris Measurements
       ↓
Feature Encoding
       ↓
Quantum State
```

---

# 🏗️ RealAmplitudes Ansatz

The first trainable quantum architecture uses:

> **RealAmplitudes**

with three repetitions.

```python
from qiskit.circuit.library import RealAmplitudes

ansatz = RealAmplitudes(
    num_qubits=num_features,
    reps=3
)
```

The number of qubits corresponds to the number of input features.

With the original four-dimensional Iris dataset:

```text
4 Features
   ↓
4 Qubits
```

---

# 🔧 COBYLA Optimizer

The quantum model is trained using the:

> **COBYLA optimizer**

```python
optimizer = COBYLA(maxiter=100)
```

COBYLA iteratively modifies the trainable parameters of the quantum circuit in order to optimize the VQC's objective function.

The overall training process is:

```text
Initial Quantum Parameters
          ↓
       VQC Output
          ↓
     Objective Function
          ↓
       COBYLA
          ↓
   Updated Parameters
          ↓
        Repeat
```

---

# 🧠 Variational Quantum Classifier

The model is created using:

```python
from qiskit_machine_learning.algorithms.classifiers import VQC

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer
)
```

The VQC combines:

* Quantum feature encoding
* Trainable quantum circuit
* Quantum sampling
* Classical optimization

into a complete classification model.

---

# 📈 Training Visualization

A callback is used to track the objective function during optimization.

This allows the project to visualize how the objective changes as training progresses.

Conceptually:

```text
Objective
   │\
   │ \
   │  \
   │   \______
   │
   └──────────────────
       Iterations
```

This provides a simple visual explanation of the optimization process.

---

# 📊 Four-Feature VQC Performance

Using all four Iris features, the project obtains:

| Model                | Training Score | Test Score |
| -------------------- | -------------: | ---------: |
| SVC                  |       **0.99** |   **0.97** |
| VQC + RealAmplitudes |       **0.87** |   **0.85** |

The quantum model successfully learns the classification task, but in this experiment its performance is lower than the classical SVC baseline.

This is an important result rather than a failure: the experiment demonstrates that a QML model can solve the problem, while also showing that **using a quantum model does not automatically imply better performance**.

---

# 📉 Dimensionality Reduction using PCA

The notebook then investigates what happens when the four Iris features are reduced to two.

The project explicitly states that this reduction is primarily for **educational purposes**.

The workflow becomes:

```text
4 Original Features
        ↓
       PCA
        ↓
2 Principal Components
        ↓
Quantum Model
```

This also makes the data easier to visualize in two dimensions.

---

# 🔵 Classical SVC with Two Features

The classical benchmark is retrained using the two PCA-derived features.

The resulting performance is:

| Model            | Training Score | Test Score |
| ---------------- | -------------: | ---------: |
| SVC — 2 Features |       **0.97** |   **0.90** |

The reduction in dimensionality results in a noticeable decrease in test performance compared with the four-feature SVC.

---

# ⚛️ VQC with Two Features

The quantum model is then rebuilt using two qubits:

```text
2 Features
   ↓
2 Qubits
```

The same general architecture is retained:

```text
ZZ Feature Map
      +
RealAmplitudes Ansatz
      +
COBYLA
      ↓
VQC
```

---

# 📊 VQC with Two Features — RealAmplitudes

The resulting performance is:

| Model                | Training Score | Test Score |
| -------------------- | -------------: | ---------: |
| VQC + RealAmplitudes |       **0.63** |   **0.58** |

This is significantly lower than both:

* Four-feature VQC
* Two-feature classical SVC

This experiment demonstrates the impact of **feature reduction on quantum model performance**.

---

# 🔥 EfficientSU2 Ansatz

The project then replaces the `RealAmplitudes` ansatz with:

> **EfficientSU2**

```python
from qiskit.circuit.library import EfficientSU2

ansatz = EfficientSU2(
    num_qubits=num_features,
    reps=3
)
```

This allows the project to investigate whether the choice of quantum circuit architecture affects classification performance.

---

# 📊 VQC with EfficientSU2

The two-feature EfficientSU2 VQC achieves:

| Model              | Training Score | Test Score |
| ------------------ | -------------: | ---------: |
| VQC + EfficientSU2 |       **0.67** |   **0.71** |

Compared with the two-feature RealAmplitudes model:

```text
RealAmplitudes → Test Score: 0.58
EfficientSU2   → Test Score: 0.71
```

the EfficientSU2 architecture performs substantially better in this experiment.

This provides an important introductory lesson:

> **The architecture of a variational quantum circuit can significantly influence model performance.**

---

# 🏆 Final Model Comparison

The notebook concludes by presenting all major experiments together.

| Model                             | Test Score | Train Score |
| --------------------------------- | ---------: | ----------: |
| **SVC — 4 Features**              |   **0.99** |    **0.97** |
| VQC — 4 Features + RealAmplitudes |       0.85 |        0.87 |
| **SVC — 2 Features**              |   **0.97** |    **0.90** |
| VQC — 2 Features + RealAmplitudes |       0.58 |        0.63 |
| VQC — 2 Features + EfficientSU2   |       0.71 |        0.67 |

---

# 🔍 Key Findings

Several useful observations emerge from the experiments.

### 1. Classical SVC Performs Extremely Well

The four-feature SVC achieves a **0.97 test score**, establishing a very strong baseline.

### 2. The Four-Feature VQC Learns the Problem

The VQC achieves a **0.85 test score**, demonstrating that a relatively small quantum model can learn the classification task.

### 3. PCA Reduces Performance

Reducing the four original features to two causes performance to decrease for both classical and quantum models.

### 4. Quantum Architecture Matters

The two-feature VQC performs differently depending on the ansatz:

```text
RealAmplitudes → 0.58 Test Score
EfficientSU2   → 0.71 Test Score
```

### 5. Quantum ≠ Automatically Better

The experiment clearly demonstrates that a quantum model does not automatically outperform a classical model simply because it uses quantum circuits.

That is an important lesson in responsible Quantum Machine Learning.

---

# 🧠 Concepts Covered

## Classical Machine Learning

* Classification
* Train/Test Split
* Feature Scaling
* Support Vector Machines
* Model Evaluation
* Dimensionality Reduction
* PCA

## Quantum Computing

* Qubits
* Quantum Circuits
* Quantum Feature Maps
* Parameterized Quantum Circuits
* Quantum Ansätze
* Quantum Sampling

## Quantum Machine Learning

* Variational Quantum Classifier
* VQC
* ZZ Feature Map
* RealAmplitudes
* EfficientSU2
* Quantum Model Optimization
* Hybrid Quantum-Classical Learning

---

# 🛠️ Technologies Used

* Python
* Qiskit
* Qiskit Machine Learning
* Qiskit Algorithms
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Jupyter Notebook
* Google Colab

---

# 📦 Installation

Install the required packages:

```bash
pip install qiskit
pip install qiskit-machine-learning
pip install qiskit-algorithms
pip install pylatexenc
pip install scikit-learn
pip install pandas numpy matplotlib seaborn
```

> **Note:** The notebook was developed around the Qiskit APIs available in its original environment. Qiskit APIs can change between versions, so reproducing the notebook exactly may require using compatible package versions.

---

# ▶️ Getting Started

## 1. Clone the Repository

```bash
git clone <repository-url>
cd Quantum-ML-Iris
```

## 2. Open the Notebook

Open:

```text
quantum_ml_model_on_iris_dataset.ipynb
```

using:

* Google Colab
* Jupyter Notebook
* JupyterLab

## 3. Run Sequentially

Execute the notebook from beginning to end.

The notebook is structured as a tutorial, so the code and explanations progressively build upon the concepts introduced earlier.

---

# 📁 Suggested Repository Structure

```text
Quantum-ML-Iris/
│
├── quantum_ml_model_on_iris_dataset.ipynb
├── README.md
└── requirements.txt
```

---

# 🎓 Learning Outcomes

After completing this project, a beginner should be able to explain:

* What Quantum Machine Learning is.
* Why classical benchmarks are important.
* How classical data can be encoded into quantum circuits.
* What a quantum feature map does.
* What a variational ansatz is.
* How a VQC performs classification.
* How classical optimizers train quantum circuits.
* Why feature scaling matters.
* What PCA does.
* How reducing features affects model performance.
* Why different quantum ansätze can produce different results.
* Why a QML model should always be compared against an appropriate classical baseline.

---

# ⚠️ Important Interpretation

This project is primarily an **educational QML experiment**, not a demonstration of quantum advantage.

The Iris dataset is:

* Small
* Clean
* Highly studied
* Relatively easy to classify
* Well suited to classical machine learning

Consequently, the fact that the classical SVC outperforms the VQC in these experiments is not surprising.

The purpose of the project is instead to demonstrate **how a quantum classifier can be constructed, trained, and evaluated on a familiar classification problem**.

The results should therefore be interpreted as a learning exercise rather than evidence for or against the practical superiority of quantum machine learning.

---

# 🚀 Possible Extensions

Once the basic notebook is understood, the project can be extended in several directions.

### 🔬 Better Quantum Models

Experiment with:

* Different feature maps
* Different ansätze
* More repetitions
* Different optimizers
* Different circuit depths

### ⚙️ Hyperparameter Optimization

Investigate:

* Number of repetitions
* Optimizer settings
* Learning/optimization parameters
* Number of qubits
* Circuit depth

### 📊 More Rigorous Evaluation

Instead of relying on a single train/test split, use:

* Cross-validation
* Multiple random seeds
* Precision
* Recall
* F1-score
* Confusion matrix

### ⚛️ Quantum Hardware

Run the VQC on actual quantum hardware and compare:

```text
Ideal Simulator
      vs
Real Quantum Hardware
```

to investigate the impact of quantum noise.

### 🤖 Classical vs Quantum Study

Extend the comparison to:

* SVM
* Logistic Regression
* Random Forest
* Classical Neural Network
* VQC

and compare their performance, parameter counts, training time, and robustness.

---

# 🤝 Contributions

Contributions are welcome, particularly those that make the project more useful as a learning resource.

Potential contributions include:

* Additional QML algorithms
* More classical baselines
* Improved visualizations
* Circuit explanations
* Hyperparameter experiments
* Quantum hardware implementations
* Beginner-friendly exercises

Feel free to fork the repository and submit a pull request.

---

# ⭐ Support the Project

If you found this project useful for learning **Quantum Machine Learning**, **Qiskit**, **Variational Quantum Classifiers**, or **Classical vs Quantum ML**, consider giving the repository a **⭐ Star**.

> **A familiar dataset, an unfamiliar paradigm — a practical introduction to Quantum Machine Learning.**
