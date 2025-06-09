from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

app = Flask(__name__)

# Indlæs og forbered data
data_path = 'Model/Matches.csv'
raw_data = pd.read_csv(data_path, low_memory=False) 

# Vi renamer kolonnerne for at undgå problemer med mellemrum
cols_to_numeric = ['HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away',
                   'HomeCorners', 'AwayCorners', 'HomeYellow', 'AwayYellow', 'FTHome', 'FTAway']
raw_data[cols_to_numeric] = raw_data[cols_to_numeric].apply(pd.to_numeric, errors='coerce')
raw_data.dropna(subset=cols_to_numeric, inplace=True)

raw_data['EloDiff'] = raw_data['HomeElo'] - raw_data['AwayElo']
raw_data['Form3Diff'] = raw_data['Form3Home'] - raw_data['Form3Away']
raw_data['Form5Diff'] = raw_data['Form5Home'] - raw_data['Form5Away']
raw_data['TotalCorners'] = raw_data['HomeCorners'] + raw_data['AwayCorners']
raw_data['TotalYellow'] = raw_data['HomeYellow'] + raw_data['AwayYellow']
raw_data['FormDiff'] = raw_data['Form5Home'] - raw_data['Form5Away']

features = ['HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away', 'EloDiff', 'FormDiff']

# Vi inkluderer kun de relevante features for modeltræning
X = raw_data[features]
y_result = raw_data['FTResult']
y_corners = raw_data['TotalCorners']
y_yellow = raw_data['TotalYellow']

# Imputer
# SimpleImputer til at håndtere manglende værdier til gennemsnit
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Encode FTResult til 0, 1, 2
# 'H' for Home Win, 'A' for Away Win, 'D' for Draw
label_encoder = LabelEncoder()
y_encoded_result = label_encoder.fit_transform(y_result)
y_encoded_result_cat = to_categorical(y_encoded_result)

# Split data
# Vi deler data i trænings- og test-sæt for både resultat, hjørnespark og gule kort
X_train, X_test, y_train_result, y_test_result = train_test_split(X_imputed, y_encoded_result_cat, test_size=0.2, random_state=42)
_, _, y_train_corners, _ = train_test_split(X_imputed, y_corners, test_size=0.2, random_state=42)
_, _, y_train_yellow, _ = train_test_split(X_imputed, y_yellow, test_size=0.2, random_state=42)

# Model til resultat (klassifikation)
# Vi bruger en simpel feedforward neural netværksmodel til at forudsige kampresultatet
model_result = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])
model_result.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_result.fit(X_train, y_train_result, epochs=10, batch_size=8, validation_split=0.1)

# Model til hjørnespark (regression)
# Vi bruger en simpel feedforward neural netværksmodel til at forudsige antallet af hjørnespark
model_corners = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1)
])
model_corners.compile(optimizer='adam', loss='mse')
model_corners.fit(X_train, y_train_corners, epochs=10, batch_size=8, validation_split=0.1)

# Model til gule kort (regression)  
# Vi bruger en simpel feedforward neural netværksmodel til at forudsige antallet af gule kort
model_yellow = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1)
])
# Vi bruger Mean Squared Error som tab for regression 
model_yellow.compile(optimizer='adam', loss='mse')
model_yellow.fit(X_train, y_train_yellow, epochs=10, batch_size=8, validation_split=0.1)

# Flask app setup
@app.route('/')
def index():
    return render_template('index.html')

## Forudsigelsesrute 
# Denne rute håndterer POST-anmodninger fra formularen og returnerer forudsigelser baseret på de indtastede holdnavne. 
@app.route('/predict', methods=['POST'])
def predict():
    hometeam = request.form['Hometeam']
    awayteam = request.form['Awayteam']

    # Tjek om holdnavne er tomme
    subset = raw_data[
        (raw_data['HomeTeam'] == hometeam) & (raw_data['AwayTeam'] == awayteam)
    ]
    # Hvis der ikke er data for det specifikke match, tjekker vi om der er data for omvendt match
    if subset.empty:
        subset = raw_data[
            ((raw_data['HomeTeam'] == hometeam) & (raw_data['AwayTeam'] == awayteam)) |
            ((raw_data['HomeTeam'] == awayteam) & (raw_data['AwayTeam'] == hometeam))
        ]

    # Hvis der ikke er data for det specifikke match, returner en besked
    if subset.empty:
        prediction_text = f"Der findes ikke data for kamp mellem {hometeam} og {awayteam}."
        confidence_text = "Ingen modeltillinger tilgængelige pga. manglende data."
        num_matches = 0
    else:
        # Forbered inputdata til forudsigelse
        # Vi bruger gennemsnittet af de relevante features for det specifikke match
        avg_input = subset[features].mean().values.reshape(1, -1)

        # Forudsig resultat, hjørnespark og gule kort
        pred_result_probs = model_result.predict(avg_input)
        pred_result_class = np.argmax(pred_result_probs)
        pred_result_label = label_encoder.inverse_transform([pred_result_class])[0]

        pred_corners = model_corners.predict(avg_input)[0][0]
        pred_yellow = model_yellow.predict(avg_input)[0][0]

        if pred_result_label == 'H':
            result_text = f"{hometeam} vinder."
        elif pred_result_label == 'A':
            result_text = f"{awayteam} vinder."
        else:
            result_text = "Uafgjort."

        # Forberedelse af forudsigelsestekst
        prediction_text = (
            f"🔮 Forudsigelse for {hometeam} vs {awayteam}:<br>"
            f"🏆 Resultat: {result_text}<br>"
            f"🔁 Hjørnespark (total): {pred_corners:.0f}<br>"
            f"🟨 Gule kort (total): {pred_yellow:.0f}"
        )
        # Forberedelse af tillidstekst
        # Vi antager, at tilliden er baseret på den maksimale sandsynlighed for resultatet
        confidence_text = (
            f"Modelens tillid til resultatet: {pred_result_probs.max():.2f}<br>"
     
        )
        num_matches = len(subset)

# Returner resultatet til HTML-skabelonen
    return render_template(
        'index.html',
        prediction_text=prediction_text,
        confidence_text=confidence_text,
        num_matches=num_matches
    )

if __name__ == '__main__':
    app.run(debug=True)
