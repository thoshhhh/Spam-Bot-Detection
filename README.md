# Spam Bot Detection

A Machine Learning based system that detects whether a message or account behavior is **spam/bot generated or legitimate**.  
This project applies Natural Language Processing (NLP) techniques and classification algorithms to identify spam patterns.

---

## Project Overview

Spam bots are commonly used on social media and messaging platforms to spread advertisements, phishing links, and malicious content.  
This project builds a **machine learning model that automatically detects spam messages** using text preprocessing and classification techniques.

The system processes input text, extracts meaningful features, and predicts whether the content is **Spam or Not Spam**.

---

## Features

- Spam message detection using Machine Learning
- Text preprocessing and cleaning
- Feature extraction using NLP techniques
- Model training and evaluation
- Simple interface for spam prediction
- Flask based web interface for testing messages

---

## Tech Stack

- Python
- Machine Learning
- Natural Language Processing (NLP)
- Scikit-learn
- Pandas
- NumPy
- Flask
- Jupyter Notebook

---

## Project Structure

```
SpamBot/
│
├── dataset/                 # Dataset used for training
├── templates/               # HTML templates for Flask UI
│   └── index.html
│
├── SpamBotDetection.ipynb   # Jupyter Notebook for model training
├── app.py                   # Flask application
├── model.pkl                # Trained ML model
├── vectorizer.pkl           # Text vectorizer
└── README.md
```

---

## Machine Learning Workflow

1. Load Dataset
2. Data Cleaning & Preprocessing
3. Text Vectorization
4. Train Machine Learning Model
5. Evaluate Model Accuracy
6. Deploy model using Flask

---

## Installation

Clone the repository

```bash
git clone https://github.com/thoshhhh/Spam-Bot-Detection.git
```

Move to the project folder

```bash
cd Spam-Bot-Detection
```

Install required libraries

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the Flask server

```bash
python app.py
```

Open the browser and go to

```
http://127.0.0.1:5000
```

Enter a message and check whether it is **Spam or Not Spam**.

---

## Example

Input:

```
Congratulations! You won a free iPhone. Click here to claim now!
```

Output:

```
Spam
```

---

## Future Improvements

- Use Deep Learning models (LSTM / Transformers)
- Improve dataset size
- Deploy the model on cloud
- Add API support
- Real-time spam detection

---

## Author

**Thoshanth Reddy Mandapati**

GitHub  
https://github.com/thoshhhh

---

## License

This project is open-source and available under the MIT License.
