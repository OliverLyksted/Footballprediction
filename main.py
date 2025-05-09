from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Indlæs og forbered data
data_path = 'Model/Matches.csv'  # Tilpas filnavnet
raw_data = pd.read_csv(data_path, low_memory=False)

# Konverter relevante kolonner til numeriske værdier
cols_to_numeric = ['HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away',
                   'HomeCorners', 'AwayCorners', 'HomeYellow', 'AwayYellow']
raw_data[cols_to_numeric] = raw_data[cols_to_numeric].apply(pd.to_numeric, errors='coerce')

# Fjern rækker med manglende værdier i de nødvendige kolonner
raw_data.dropna(subset=cols_to_numeric + ['FTResult'], inplace=True)

# Funktion til at udregne totaler
raw_data['TotalCorners'] = raw_data['HomeCorners'] + raw_data['AwayCorners']
raw_data['TotalYellow'] = raw_data['HomeYellow'] + raw_data['AwayYellow']

# Definér features og targets
features = ['HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away']

X = raw_data[features]
y_result = raw_data['FTResult']
y_corners = raw_data['TotalCorners']
y_yellow = raw_data['TotalYellow']

# Imputer manglende værdier hvis nødvendigt
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Split og træn modeller
X_train, X_test, y_train_result, y_test_result = train_test_split(X_imputed, y_result, test_size=0.2, random_state=42)
_, _, y_train_corners, _ = train_test_split(X_imputed, y_corners, test_size=0.2, random_state=42)
_, _, y_train_yellow, _ = train_test_split(X_imputed, y_yellow, test_size=0.2, random_state=42)

model_result = RandomForestClassifier(random_state=42)
model_result.fit(X_train, y_train_result)

model_corners = RandomForestClassifier(random_state=42)
model_corners.fit(X_train, y_train_corners)

model_yellow = RandomForestClassifier(random_state=42)
model_yellow.fit(X_train, y_train_yellow)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hometeam = request.form['Hometeam']
    awayteam = request.form['Awayteam']

    # Find alle kampe mellem de to hold
    subset = raw_data[
        ((raw_data['HomeTeam'] == hometeam) & (raw_data['AwayTeam'] == awayteam)) |
        ((raw_data['HomeTeam'] == awayteam) & (raw_data['AwayTeam'] == hometeam))
    ]

    if subset.empty:
        prediction_text = f"Der findes ikke data for kamp mellem {hometeam} og {awayteam}."
    else:
        avg_input = subset[features].mean().values.reshape(1, -1)

        pred_result = model_result.predict(avg_input)[0]
        pred_corners = model_corners.predict(avg_input)[0]
        pred_yellow = model_yellow.predict(avg_input)[0]

        if pred_result == 'H':
            result_text = f"{hometeam} vinder."
        elif pred_result == 'A':
            result_text = f"{awayteam} vinder."
        else:
            result_text = "Uafgjort."

        prediction_text = (
            f"🔮 Forudsigelse for {hometeam} vs {awayteam}:<br>"
            f"🏆 Resultat: {result_text}<br>"
            f"🔁 Hjørnespark (total): {int(pred_corners)}<br>"
            f"🟨 Gule kort (total): {int(pred_yellow)}"
        )
        confidence_text = (
            f"Modelens tillid til resultatet: {model_result.predict_proba(avg_input)[0][1]:.2f}<br>"
            f"Modelens tillid til hjørnespark: {model_corners.predict_proba(avg_input)[0][1]:.2f}<br>"
            f"Modelens tillid til gule kort: {model_yellow.predict_proba(avg_input)[0][1]:.2f}"
        )
        num_matches = len(subset)

    return render_template(
    'index.html',
    prediction_text=prediction_text,
    confidence_text=confidence_text,
    num_matches=num_matches
)

if __name__ == '__main__':
    app.run(debug=True)
