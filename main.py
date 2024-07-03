import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords


app = Flask(__name__)
data = pd.read_csv(r'C:\Users\Administrator\sentimentanalysis\dataset\flipkart_data.csv')
# Load the trained model
model = pickle.load(open('flipkart.pkl', 'rb'))
def preprocess_text(text_data):
    preprocessed_text = []
    for sentence in text_data:
        # Removing punctuations
        sentence = re.sub(r'[^\w\s]', '', str(sentence))
        # Converting lowercase and removing stopwords
        preprocessed_text.append(' '.join(token.lower() for token in nltk.word_tokenize(sentence)
                                          if token.lower() not in stopwords.words('english')))
    return preprocessed_text

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(data['review'] ).toarray()


@app.route('/predict', methods=['POST'])
def predict():
    # Get review text from form
    review_text = request.form.get('review')

    # Preprocess the input review (if needed)
    processed_review = preprocess_text([review_text])  # Implement preprocess_text function

    # Vectorize the preprocessed review (if needed)
    input_vector = cv.transform(processed_review).toarray()  # Implement vectorizer if needed

    # Predict sentiment using the loaded model
    prediction = model.predict(input_vector)[0]

    # Determine sentiment label
    sentiment_label = 'Positive' if prediction == 1 else 'Negative'

    return render_template('result.html', review=review_text, prediction=sentiment_label)


if __name__ == '__main__':
    app.run(debug=True)