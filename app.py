from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64

app = Flask(_name_)

df = pd.DataFrame()

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Squareness'] = df['Values'].rolling(window=3).mean() - df['Values'].shift(1)
    df['Rolling_Mean'] = df['Values'].rolling(window=3).mean()
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['DayOfWeek'] = df.index.dayofweek
    df['Lag_1'] = df['Values'].shift(1)
    df['Lag_2'] = df['Values'].shift(2)
    df.dropna(inplace=True)
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_data', methods=['POST'])
def load_data():
    global df
    file = request.files['file']
    if file:
        df = pd.read_excel(file)
        df = preprocess_data(df)
        return jsonify({"message": "Data Loaded and Preprocessed"})
    return jsonify({"message": "Failed to load data"}), 400

@app.route('/predict_deviation', methods=['POST'])
def predict_deviation():
    global df
    if df.shape[0] == 0:
        return jsonify({"message": "No data available for prediction"}), 400

    try:
        n_parts_per_day = int(request.form['n_parts_per_day'])
        if n_parts_per_day <= 0:
            raise ValueError
    except ValueError:
        return jsonify({"message": "Please enter a valid positive integer for the number of parts"}), 400

    X = df[['Rolling_Mean', 'Month', 'Day', 'DayOfWeek', 'Lag_1', 'Lag_2', 'Squareness']]
    y = df['Values']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
    model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df.index, df['Values'], label='Actual Values')
    ax.plot(df.index[len(df) - len(y_pred):], y_pred, label='Predicted Values')
    ax.axhline(y=0.033, color='r', linestyle='--', label='Threshold')
    ax.legend()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return jsonify({"rmse": rmse, "plot_url": f"data:image/png;base64,{plot_url}"})

@app.route('/predict_future_deviation', methods=['POST'])
def predict_future_deviation():
    global df
    future_date = df.index[-1]
    exceeded_part = None
    try:
        n_parts_per_day = int(request.form['n_parts_per_day'])
        if n_parts_per_day <= 0:
            raise ValueError
    except ValueError:
        return jsonify({"message": "Please enter a valid positive integer for the number of parts"}), 400

    best_model = request.form['best_model']

    results = []

    for part in range(1, n_parts_per_day + 1):
        future_features = pd.DataFrame({
            'Rolling_Mean': [df['Rolling_Mean'].iloc[-3:].mean()],
            'Month': [future_date.month],
            'Day': [future_date.day],
            'DayOfWeek': [future_date.dayofweek],
            'Lag_1': [df['Values'].iloc[-1]],
            'Lag_2': [df['Values'].iloc[-2]],
            'Squareness': [df['Squareness'].iloc[-1]]
        }).ffill().bfill()

        future_pred = best_model.predict(future_features)[0]
        future_date += pd.Timedelta(days=1 / n_parts_per_day)
        results.append({"part": part, "prediction": future_pred})

        if future_pred > 0.066 and exceeded_part is None:
            exceeded_part = part

    if exceeded_part is not None:
        deviation_date = future_date
        return jsonify({"message": f"Deviation exceeds 0.066 on {deviation_date.strftime('%Y-%m-%d')} "
                                   f"(Day: {deviation_date.strftime('%A')}) with predicted value: {future_pred:.4f}"})
    else:
        return jsonify({"message": "Deviation did not exceed 0.066 in the specified parts."})

if _name_ == '_main_':
    app.run(debug=True)