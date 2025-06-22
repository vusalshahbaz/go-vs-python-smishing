# Phishing SMS Detection: Go vs Python

This project compares the performance of three machine learning models implemented in **Go** and **Python** to detect **phishing SMS messages**.

## 🧠 Machine Learning Algorithms Used

The following algorithms were trained and evaluated using the same dataset and preprocessing:

- **Naive Bayes**
- **Logistic Regression**
- **Decision Trees**

Each model was implemented and tested in both **Go** and **Python** environments to compare not only the model performance but also application-level efficiency.

---

## 📊 Model Metrics

| **Metric**   | **Go** | **Python** |
|--------------|--------|------------|
| Accuracy     | 0.95   | 0.97       |
| Precision    | 0.92   | 0.97       |
| Recall       | 0.85   | 0.87       |
| F1-Score     | 0.88   | 0.92       |

> **Description:**  
> This table shows the classification performance metrics of the models in both Go and Python implementations. While Python achieves slightly higher scores, Go remains competitive with excellent performance.

---

## ⚙️ App Metrics

| **App Metric**        | **Description**                                                              | **Go**     | **Python** |
|-----------------------|-------------------------------------------------------------------------------|------------|------------|
| **Image Size**        | Size of the compiled application or container image.                         | 388 MB     | 1.26 GB    |
| **Memory Usage**      | Peak memory usage during inference.                                          | 104 MB     | 325 MB     |
| **CPU**               | Average CPU usage during prediction (in CPU cores).                          | 0.12 cores | 0.108 cores|
| **Prediction Duration** | Time taken to make a single prediction.                                      | 10 ms      | 9 ms       |

> **Description:**  
> Despite Python being slightly faster in prediction, the Go implementation excels in memory usage and produces significantly smaller binaries — making it more efficient for deployment in resource-constrained environments.

---

## ✅ Conclusion

- **Python** delivers slightly higher accuracy and precision in ML model metrics.
- **Go** provides a lightweight and resource-efficient alternative, especially suitable for embedded or cloud-native deployments.

---