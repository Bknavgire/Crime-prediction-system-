from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Sample Dataset
crime_data = {
    'Year': [2018, 2019, 2020, 2021, 2022, 2023],
    'CrimeRate': [120, 135, 150, 170, 180, 200]
}

# Convert into DataFrame
crime_df = pd.DataFrame(crime_data)

# Training Data
X = crime_df[['Year']]
y = crime_df['CrimeRate']

# Train Model
model = LinearRegression()
model.fit(X, y)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        year = int(request.form['year'])
        prediction = model.predict([[year]])[0]
        prediction = round(prediction, 2)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
