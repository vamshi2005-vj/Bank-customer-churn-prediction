from flask import Flask, request, render_template, send_file
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load trained model
model = joblib.load('churn_predict_model.pkl')

# Load sample data for scaler fitting
data = pd.read_csv('Churn_Modelling.csv')
data = data.drop(['RowNumber', 'Surname', 'CustomerId'], axis=1)
data = pd.get_dummies(data, drop_first=True)
X = data.drop('Exited', axis=1)

scaler = StandardScaler()
scaler.fit(X)

# Global for saving predictions
predictions_df = pd.DataFrame()

@app.route('/', methods=['GET', 'POST'])
def index():
    global predictions_df
    error = None
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename.endswith('.csv'):
            error = "Please upload a valid CSV file (.csv only)."
            return render_template('index.html', predictions=None, error=error)

        try:
            df = pd.read_csv(file)
            original_df = df.copy()

            required_features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                                 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
            missing = [col for col in required_features if col not in df.columns]
            if missing:
                error = f"Missing required columns: {', '.join(missing)}"
                return render_template('index.html', predictions=None, error=error)

            df = pd.get_dummies(df, drop_first=True)

            for col in (set(X.columns) - set(df.columns)):
                df[col] = 0
            df = df[X.columns]

            df_scaled = scaler.transform(df)
            predictions = model.predict(df_scaled)

            original_df['Exited'] = predictions
            predictions_df = original_df[original_df['Exited'] == 1]

            total = len(original_df)
            at_risk = sum(predictions)
            safe = total - at_risk

            return render_template('index.html', predictions=predictions_df,
                                   total=total, at_risk=at_risk, safe=safe)

        except Exception as e:
            error = f"Error during file processing: {str(e)}"
            return render_template('index.html', predictions=None, error=error)

    return render_template('index.html', predictions=None, error=error)

@app.route('/download')
def download():
    global predictions_df
    file_path = 'at_risk_customers.xlsx'
    predictions_df.to_excel(file_path, index=False)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
