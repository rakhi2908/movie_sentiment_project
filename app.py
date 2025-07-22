from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data.get("review")
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)[0]
    return jsonify({'sentiment': prediction})

if __name__ == "__main__":
    app.run(debug=True)
