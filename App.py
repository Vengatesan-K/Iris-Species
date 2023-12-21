from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib as joblib
import os

model = joblib.load('best_log_model.pkl')

app = Flask(__name__)

IMG_FOLDER = os.path.join('Styling', 'IMG')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        sepal_length = request.form['sepal_length']
        sepal_width =  request.form['sepal_width']
        petal_length = request.form['petal_length']
        petal_width = request.form['petal_width']
        data = np.array([[sepal_length,sepal_width,petal_length,petal_width]], dtype = float)
        result = model.predict(data)
        image = result[0] + '.jpg'
        image = os.path.join(app.config['UPLOAD_FOLDER'], image)
        return render_template('index.html', result=result[0],image=image)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=8080)

