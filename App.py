from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
#loading the model
model = pickle.load(open('iris_best_model.sav', 'rb'))

@app.route('/')
def home():
    result = ""
    return render_template('index.html', result=result)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width =  float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        if result == 0:
            result = 'Iris Setosa'
        elif result == 1:
            result = 'Iris Versicolour'
        elif result == 2:
            result = 'Iris Virginica'
        else:
            result = 'Error'
        return render_template('index.html', result=result)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

