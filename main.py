import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Indlæs data
df = pd.read_csv("Matches.csv")

# Fjern rækker uden scores
df = df.dropna(subset=["home_score", "away_score"])

# Lav target-kolonne
def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'H'
    elif row['home_score'] < row['away_score']:
        return 'A'
    else:
        return 'D'

df['match_result'] = df.apply(get_result, axis=1)

# Encode holdnavne
from sklearn.preprocessing import LabelEncoder
home_encoder = LabelEncoder()
away_encoder = LabelEncoder()

df['home_team_encoded'] = home_encoder.fit_transform(df['home_team'].astype(str))
df['away_team_encoded'] = away_encoder.fit_transform(df['away_team'].astype(str))

# Træk årstal ud
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df = df.dropna(subset=['year'])

# Features og label
X = df[['home_team_encoded', 'away_team_encoded', 'year']]
y = df['match_result']

# Split og træning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluering
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Gem model og encodere
joblib.dump(model, "model/match_predictor.pkl")
joblib.dump(home_encoder, "model/home_encoder.pkl")
joblib.dump(away_encoder, "model/away_encoder.pkl")
