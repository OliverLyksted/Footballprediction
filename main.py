//pip install flask pandas scikit-learn matplotlib seaborn in terminal


from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the dataset and train the model when the app starts
data_path = 'Footballprediction/Matches.zip'  # Adjust path if needed
data = pd.read_csv(data_path)

# Feature selection (customize according to your dataset)
X = data.iloc[:, :-1]  # All columns except the last one (features)
y = data.iloc[:, -1]   # Last column (target: match result)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model to a file using pickle (this is useful for reloading the model later)
if not os.path.exists('model'):
    os.mkdir('model')
model_filename = 'model/football_prediction_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)

@app.route('/')
def home():
    return render_template('index.html')  # This will render the homepage

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data from user input
    try:
        features = [float(request.form.get('feature1')), float(request.form.get('feature2'))]  # Adjust for your features
        # Load the model from the pickle file
        with open(model_filename, 'rb') as model_file:
            model = pickle.load(model_file)
        
        # Make prediction
        prediction = model.predict([features])

        # Return the prediction result to the user
        return render_template('index.html', prediction_text=f'The predicted result is: {prediction[0]}')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
