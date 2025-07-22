import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

data = {
    'text': ["I loved the movie", "It was fantastic", "Worst film ever", "I hated it", "Amazing experience", "Terrible plot", "good", "nice", "great", "needs improvement"],
    'label': ["positive", "positive", "negative", "negative", "positive", "negative", "positive", "positive", "positive", "negative"]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
