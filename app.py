from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

with open("stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True, vocabulary=pickle.load(open("tfidfvectoizer.pkl", "rb")))
model = pickle.load(open("LinearSVCTuned.pkl", 'rb'))

# Define a function to map predictions to categories
def get_bullying_category(prediction):
    if prediction == 0:
        return "Not Bullying"
    elif prediction == 1:
        return "Bullying due to Ethnicity"
    elif prediction == 2:
        return "Bullying due to Age"
    elif prediction == 3:
        return "Bullying due to Religion"
    else:
        return "Other Bullying"

@app.route('/', methods=['GET', 'POST'])
def index():
    category = None
    if request.method == 'POST':
        user_input = request.form['text']
        transformed_input = vectorizer.transform([user_input])  # Use transform instead of fit_transform
        prediction = model.predict(transformed_input)[0]
        category = get_bullying_category(prediction)  # Get the bullying category based on prediction
    
    return render_template('index.html', category=category)

if __name__ == '__main__':
    app.run(debug=True)
