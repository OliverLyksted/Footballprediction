from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Indlæs og forbered data
data_path = 'Model/Matches.csv'
raw_data = pd.read_csv(data_path, low_memory=False)

# Konverter relevante kolonner
cols_to_numeric = ['HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away',
                   'HomeCorners', 'AwayCorners', 'HomeYellow', 'AwayYellow', 'FTHome', 'FTAway']
raw_data[cols_to_numeric] = raw_data[cols_to_numeric].apply(pd.to_numeric, errors='coerce')

# Fjern rækker med manglende data
raw_data.dropna(subset=cols_to_numeric, inplace=True)

# Ekstra features
raw_data['EloDiff'] = raw_data['HomeElo'] - raw_data['AwayElo']
raw_data['Form3Diff'] = raw_data['Form3Home'] - raw_data['Form3Away']
raw_data['Form5Diff'] = raw_data['Form5Home'] - raw_data['Form5Away']
raw_data['TotalCorners'] = raw_data['HomeCorners'] + raw_data['AwayCorners']
raw_data['TotalYellow'] = raw_data['HomeYellow'] + raw_data['AwayYellow']

<<<<<<< HEAD
# Tilføj forskels-features
raw_data['EloDiff'] = raw_data['HomeElo'] - raw_data['AwayElo']
raw_data['FormDiff'] = raw_data['Form5Home'] - raw_data['Form5Away']

# Opdater features
features = ['HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away', 'EloDiff', 'FormDiff']


=======
# Features og targets
features = ['EloDiff', 'Form3Diff', 'Form5Diff']
>>>>>>> 9bb8dad69b25c5194bec9686184e4b7c3e471059
X = raw_data[features]
y_goals = raw_data[['FTHome', 'FTAway']]
y_corners = raw_data['TotalCorners']
y_yellow = raw_data['TotalYellow']

# Imputer manglende værdier
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Train-test splits
X_train, X_test, y_train_goals, y_test_goals = train_test_split(X_imputed, y_goals, test_size=0.2, random_state=42)
_, _, y_train_corners, _ = train_test_split(X_imputed, y_corners, test_size=0.2, random_state=42)
_, _, y_train_yellow, _ = train_test_split(X_imputed, y_yellow, test_size=0.2, random_state=42)

<<<<<<< HEAD
model_result = RandomForestClassifier(random_state=42, class_weight='balanced')
model_result.fit(X_train, y_train_result)
=======
# Modeller
model_goals = MultiOutputRegressor(RandomForestRegressor(random_state=42))
model_goals.fit(X_train, y_train_goals)
>>>>>>> 9bb8dad69b25c5194bec9686184e4b7c3e471059

model_corners = RandomForestClassifier(random_state=42, class_weight='balanced')
model_corners.fit(X_train, y_train_corners)

model_yellow = RandomForestClassifier(random_state=42, class_weight='balanced')
model_yellow.fit(X_train, y_train_yellow)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hometeam = request.form['Hometeam']
    awayteam = request.form['Awayteam']

<<<<<<< HEAD
    # Find kun kampe hvor hjemme- og udehold er i samme roller som input
=======
>>>>>>> 9bb8dad69b25c5194bec9686184e4b7c3e471059
    subset = raw_data[
        (raw_data['HomeTeam'] == hometeam) & (raw_data['AwayTeam'] == awayteam)
    ]

    # Fallback til alle indbyrdes kampe hvis der ikke findes nogen i korrekt opstilling
    if subset.empty:
        subset = raw_data[
            ((raw_data['HomeTeam'] == hometeam) & (raw_data['AwayTeam'] == awayteam)) |
            ((raw_data['HomeTeam'] == awayteam) & (raw_data['AwayTeam'] == hometeam))
        ]

    if subset.empty:
        prediction_text = f"Der findes ikke data for kamp mellem {hometeam} og {awayteam}."
        confidence_text = "Ingen modeltillinger tilgængelige pga. manglende data."
        num_matches = 0
    else:
        avg_input = subset[features].mean().values.reshape(1, -1)

        # Predict goals
        predicted_goals = model_goals.predict(avg_input)[0]
        pred_home_goals = int(round(predicted_goals[0]))
        pred_away_goals = int(round(predicted_goals[1]))

        # Derive result
        if pred_home_goals > pred_away_goals:
            result_text = f"{hometeam} vinder."
        elif pred_away_goals > pred_home_goals:
            result_text = f"{awayteam} vinder."
        else:
            result_text = "Uafgjort."

        # Predict corners and yellow cards
        pred_corners = model_corners.predict(avg_input)[0]
        pred_yellow = model_yellow.predict(avg_input)[0]

        prediction_text = (
            f"🔮 Forudsigelse for {hometeam} vs {awayteam}:<br>"
            f"🏆 Resultat: {result_text} ({hometeam} {pred_home_goals} - {pred_away_goals} {awayteam})<br>"
            f"🔁 Hjørnespark (total): {int(pred_corners)}<br>"
            f"🟨 Gule kort (total): {int(pred_yellow)}"
        )
<<<<<<< HEAD
        confidence_text = (
            f"Modelens tillid til resultatet: {model_result.predict_proba(avg_input).max():.2f}<br>"
            f"Modelens tillid til hjørnespark: {model_corners.predict_proba(avg_input).max():.2f}<br>"
            f"Modelens tillid til gule kort: {model_yellow.predict_proba(avg_input).max():.2f}"
        )
=======
        confidence_text = "Resultatet er baseret på forudsagte mål afrundet til hele tal."
>>>>>>> 9bb8dad69b25c5194bec9686184e4b7c3e471059
        num_matches = len(subset)

    return render_template(
        'index.html',
        prediction_text=prediction_text,
        confidence_text=confidence_text,
        num_matches=num_matches
    )


if __name__ == '__main__':
    app.run(debug=True)
