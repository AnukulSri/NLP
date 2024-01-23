# Creating a flask application for spam classifier...

from flask import Flask, render_template, request
import pandas as pd
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the pre-trained model and necessary data
df = pd.read_csv('C:\\Users\\anuku\\OneDrive\\Desktop\\NLP\\SMSSpamCollection', sep='\t', names=["label", "message"])
corpus = []
l = df['message'].to_string()
sentence = nltk.sent_tokenize(l)
ps = PorterStemmer()
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

tv = CountVectorizer(max_features=5000)
x = tv.fit_transform(corpus).toarray()
y = pd.get_dummies(df['label'])
y = y.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
spam_detect_model = MultinomialNB().fit(X_train, Y_train)

# Function to predict whether a message is spam or not
def predict_spam(message):
    review = re.sub('[^a-zA-Z]', ' ', message)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    test_corpus = [review]
    test_x = tv.transform(test_corpus).toarray()
    prediction = spam_detect_model.predict(test_x)
    return prediction[0]

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        prediction = predict_spam(message)
        if prediction == 1:
            result = 'Spam'
        else:
            result = 'Not Spam'
        return render_template('result.html', message=message, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
