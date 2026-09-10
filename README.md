# Student Performance Prediction & Cloud Deployment

An end-to-end Machine Learning pipeline designed to predict student Math Scores based on demographic, socioeconomic, and academic features[cite: 1, 2]. The project includes robust data preprocessing, model selection, custom exception handling, and is deployed as a web application on AWS Elastic Beanstalk[cite: 1, 2].

## Project Overview
Predicting academic performance helps educators identify students who may need additional support early on. This project takes various demographic and background features, processes them through a modular ML pipeline, evaluates multiple regression models, and serves real-time predictions via a Flask web interface[cite: 1, 2].

## Key Features
* **Modular Architecture:** Built with custom exception handling and logging for maintainability and debugging[cite: 1, 2].
* **Robust Data Preprocessing:** Applies One-Hot Encoding for categorical data and standard scaling for numerical features via serialized preprocessors (`preprocessor.pkl`)[cite: 1, 2].
* **Rigorous Model Evaluation:** Trained and compared baseline and ensemble models (Linear Regression, Ridge, Lasso, and Random Forest), achieving an **$R^2$ score of 0.882** and a **4.2 Mean Absolute Error (MAE)** using an optimized Random Forest Regressor[cite: 1, 2].
* **Cloud Deployment:** Integrated with a Flask web application and deployed to production on **AWS Elastic Beanstalk**[cite: 1, 2].

## Tech Stack
* **Language:** Python[cite: 1, 2]
* **Libraries:** Scikit-learn, Pandas, NumPy[cite: 1, 2]
* **Web Framework:** Flask[cite: 1, 2]
* **Cloud Platform:** AWS Elastic Beanstalk[cite: 1, 2]

## Project Structure
```text
├── artifacts/             # Serialized models and preprocessors (model.pkl, preprocessor.pkl)
├── src/                   # Source code for pipeline, components, logging, and exceptions
├── templates/             # HTML templates for the web interface
├── app.py                 # Flask application entry point
├── requirements.txt       # Project dependencies
└── README.md


Setup and Installation
Clone the repository:

Bash
git clone [https://github.com/your-username/student-performance-prediction.git](https://github.com/your-username/student-performance-prediction.git)
cd student-performance-prediction
Create a virtual environment and activate it:

Bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
Install dependencies:

Bash
pip install -r requirements.txt
Run the application locally:

Bash
python app.py
Open your browser:
Navigate to http://127.0.0.1:5000/ to test real-time predictions.