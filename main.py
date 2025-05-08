from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

# Initialize the Flask app
app = Flask(__name__)

# Load original dataset
data_path = 'Model/Matches.csv'
raw_data = pd.read_csv(data_path)  # Behold den oprindelige
# Lav dummy-version til model-træning
data = pd.get_dummies(raw_data, drop_first=True)

# Feature selection (customize according to your dataset)
X = data.iloc[:, 0:42]  # All columns except the last one (features)
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
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        hometeam = request.form.get('Hometeam')
        awayteam = request.form.get('Awayteam')

        # Find rækker i datasættet hvor disse hold spillede mod hinanden
        # Brug raw_data her i stedet for data
        match = raw_data[(raw_data['HomeTeam'] == hometeam) & (raw_data['AwayTeam'] == awayteam)]



        if match.empty:
            return render_template('index.html', prediction_text=f'Ingen kampdata fundet for {hometeam} vs {awayteam}')

        # Forbered inputdata som modelen kan bruge
        match_encoded = pd.get_dummies(match, drop_first=True)
        # Tilføj manglende kolonner (hvis nødvendigt)
        missing_cols = set(X.columns) - set(match_encoded.columns)
        for col in missing_cols:
            match_encoded[col] = 0
        match_encoded = match_encoded[X.columns]  # Sørg for korrekt kolonnerækkefølge

        # Load modellen
        with open(model_filename, 'rb') as model_file:
            model = pickle.load(model_file)

        # Forudsig resultat
        prediction = model.predict(match_encoded)

        return render_template('index.html', prediction_text=f'Det forudsagte resultat for {hometeam} vs {awayteam} er: {prediction[0]}')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Fejl: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
