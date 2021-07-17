import pandas as pd
import pickle
from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
rnd_model=pickle.load(open('rnd_reg.pkl','rb'))

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['GET','post'])
def predict():   
    GRE_Score = int(request.form['GRE Score'])
    TOEFL_Score = int(request.form['TOEFL Score'])
    University_Rating = int(request.form['University Rating'])
    SOP = float(request.form['SOP'])
    LOR = float(request.form['LOR'])
    CGPA = float(request.form['CGPA'])
    Research = int(request.form['Research'])
    final_data = pd.DataFrame([[GRE_Score, TOEFL_Score, University_Rating, SOP, LOR, CGPA, Research]])
    predict = rnd_model.predict(final_data)
    output = predict[0]
    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
    