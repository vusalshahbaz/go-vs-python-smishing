# Machine Learning: Go vs Python Comparison

This project compares the performance of machine learning implementations in **Go** and **Python** across three different applications. We developed three apps to compare implementations of Golang and Python:

1. **Spam Detection** - Using UCI Spam SMS Collection dataset
2. **Phishing Email Detection** - Email phishing detection system
3. **Text Classification** - Large-scale text classification for 50k IMDB dataset

## 🧠 Machine Learning Algorithms Used

The following algorithms were trained and evaluated using the same dataset and preprocessing:

- **Naive Bayes**
- **Logistic Regression**
- **Decision Trees**

Each model was implemented and tested in both **Go** and **Python** environments to compare not only the model performance but also application-level efficiency.

---

## 📊 Results

### Spam Detection Model Metrics

| | **Accuracy** | **Precision (macro)** | **Recall (macro)** | **F1** | **Training Time (seconds)** | **Test Time (seconds)** |
|---|---|---|---|---|---|---|
| **Go** | 96,59 | 98,1 | 87,24 | 91,72 | 0.4 | < 0.1 |
| **Python** | 98,47 | 99,13 | 94,29 | 96,53 | < 0.1 | < 0.1 |

### Email Phishing Detection

| | **Accuracy** | **Precision (macro)** | **Recall (macro)** | **F1** | **Training Time (seconds)** | **Test Time (seconds)** |
|---|---|---|---|---|---|---|
| **Go** | 98,77 | 99,28 | 96,19 | 97,66 | 2,6 | 0.2 |
| **Python** | 99,47 | 99,24 | 98,8 | 99,02 | 0,4 | < 0.1 |

### Text Classification

| | **Accuracy** | **Precision** | **Recall** | **F1** | **Training Time (seconds)** | **Test Time (seconds)** |
|---|---|---|---|---|---|---|
| **Go** | 80,63 | 81,89 | 80,63 | 80,43 | 46 | 67 |
| **Python** | 87,28 | 87,31 | 87,28 | 87,27 | 10 | < 0,1 |

---

## ⚙️ App Metrics

| **App Metric**          | **Description**                                      | **Go**     | **Python**  |
|-------------------------|------------------------------------------------------|------------|-------------|
| **Image Size**          | Size of the compiled application or container image. | 388 MB     | 1.26 GB     |
| **Memory Usage**        | Peak memory usage during inference.                  | 104 MB     | 325 MB      |
| **CPU**                 | Average CPU usage during prediction (in CPU cores).  | 0.12 cores | 0.108 cores |
| **Prediction Duration** | Time taken to make a single prediction.              | 10 ms      | 9 ms        |

> **Description:**  
> Despite Python being slightly faster in prediction, the Go implementation excels in memory usage and produces significantly smaller binaries — making it more efficient for deployment in resource-constrained environments.

---

## ✅ Conclusion

While Python shows slightly better performance in model accuracy and prediction speed, this advantage is largely due to its mature machine learning ecosystem:
- Most popular Python libraries (like NumPy, scikit-learn, pandas) are built as wrappers around highly optimized C/C++ code, enabling fast numerical operations.
- Python benefits from a large and active community, offering a wide range of ready-to-use ML tools, pre-trained models, and educational resources.
-  As a result, tasks like text vectorization (e.g., TF-IDF), model training, and evaluation are straightforward and efficient in Python.

On the other hand, Go (Golang):
- Has limited support for advanced machine learning out of the box.
- Lacks native high-level ML libraries and often requires custom implementations or bindings to external C libraries.
- Is still an emerging choice for ML but excels in performance, concurrency, and deployment (especially for building fast, lightweight, and statically compiled services).

> 🧠 Summary:
> Python is better suited for fast prototyping and leveraging existing ML solutions, while Go is ideal for production-level systems where performance, memory efficiency, and deployment simplicity matter most.
---