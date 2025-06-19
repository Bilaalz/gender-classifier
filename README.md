# Gender Classifier with Scikit-learn

This project compares four different machine learning models to classify gender based on height, weight, and shoe size. It was built using Python and scikit-learn as a hands-on way to explore supervised classification and deepen my understanding of machine learning fundamentals.

## 🔍 What the Project Does

The script trains and evaluates the following models on a small custom dataset:

- **Decision Tree Classifier**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Gaussian Naive Bayes**

Each model predicts the gender of a new sample `[190cm, 70kg, 43 shoe size]` and reports its accuracy on the training data.

## 📈 Sample Output
Decision Tree Prediction: male | Accuracy: 1.0
KNN Prediction: male | Accuracy: 0.91
SVM Prediction: female | Accuracy: 0.82
Naive Bayes Prediction: male | Accuracy: 0.91

## 📚 What I Learned

- The structure of a supervised machine learning pipeline
- How different classifiers "think" and behave on the same dataset
- The importance of comparing models and not relying on one algorithm
- How to evaluate models using accuracy
- Realized that even with small datasets, model behavior can vary significantly

## 💡 Next Steps

To take this project further, I plan to:
- Introduce train/test splitting to measure real-world generalization accuracy
-	Expand the dataset to improve model performance and reduce overfitting
-	Scrape real-world data or find a gender-based dataset from platforms like Kaggle
-	Wrap the models into a web app using Flask or Streamlit for user-friendly predictions
-	Visualize decision boundaries and model behavior with graphs

## ⚙️ How to Run

Clone the repository and run the script using Python:

```bash
python main.py
