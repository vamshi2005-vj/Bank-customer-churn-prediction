from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('churn_predict_model.pkl')

# Load and preprocess the data to fit the scaler
data = pd.read_csv('Churn_Modelling.csv')
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
data = pd.get_dummies(data, drop_first=True)
X = data.drop('Exited', axis=1)
sc = StandardScaler()
sc.fit(X)

# Feature names for scaling
feature_names = X.columns

# Path for the Excel file
excel_file_path = 'risk_customers.xlsx'

def append_to_excel(df, file_path):
    if os.path.exists(file_path):
        # Load existing data and append new data
        with pd.ExcelWriter(file_path, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    else:
        # If the file does not exist, write the new data to a new file
        df.to_excel(file_path, index=False)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ""
    if request.method == 'POST':
        # Get form data
        customer_id = int(request.form['CustomerID'])
        name = request.form['Name']
        p1 = int(request.form['CreditScore'])
        p2 = int(request.form['Age'])
        p3 = int(request.form['Tenure'])
        p4 = float(request.form['Balance'])
        p5 = int(request.form['NumOfProducts'])
        p6 = int(request.form['HasCrCard'])
        p7 = int(request.form['IsActiveMember'])
        p8 = float(request.form['EstimatedSalary'])
        p9 = int(request.form['Geography'])
        p10 = int(request.form['Gender'])
        
        # Determine geography values
        Geography_Germany = 1 if p9 == 1 else 0
        Geography_Spain = 1 if p9 == 2 else 0
        Geography_France = 1 if p9 == 3 else 0
        
        # Create a DataFrame with the correct feature names
        input_data = pd.DataFrame([[p1, p2, p3, p4, p5, p6, p7, p8, Geography_Germany, Geography_Spain, p10]],
                                  columns=feature_names)
        input_scaled = sc.transform(input_data)
        result = model.predict(input_scaled)

        prediction = "The Customer is at Risk" if result == 1 else "The Customer is at No Risk"

        # If customer is at risk (result == 1), store in Excel
        if result == 1:
            risk_data = pd.DataFrame([{
                'Customer ID': customer_id,
                'Name': name,
                'Credit Score': p1,
                'Age': p2,
                'Tenure': p3,
                'Balance': p4,
                'Number of Products': p5,
                'Has Credit Card': p6,
                'Is Active Member': p7,
                'Estimated Salary': p8,
                'Geography': p9,
                'Gender': p10
            }])
            append_to_excel(risk_data, excel_file_path)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
#changed!!