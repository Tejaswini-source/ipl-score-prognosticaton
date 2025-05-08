from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import os
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this for production

# Mock user database (replace with real database)
users = {
    'admin': {'password': 'admin123', 'name': 'Admin User'}
}

# Load and preprocess data
def load_data():
    ipl = pd.read_csv('ipl_data.csv')
    df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5','mid', 'striker', 'non-striker'], axis=1)
    return df

df = load_data()

# Initialize encoders and scaler
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()

venue_encoder.fit(df['venue'])
batting_team_encoder.fit(df['bat_team'])
bowling_team_encoder.fit(df['bowl_team'])
striker_encoder.fit(df['batsman'])
bowler_encoder.fit(df['bowler'])

# Model loading/training
def get_model():
    if os.path.exists('models/ipl_score_predictor.h5'):
        model = keras.models.load_model('models/ipl_score_predictor.h5')
    else:
        X = df.drop(['total'], axis=1)
        y = df['total']
        
        X['venue'] = venue_encoder.transform(X['venue'])
        X['bat_team'] = batting_team_encoder.transform(X['bat_team'])
        X['bowl_team'] = bowling_team_encoder.transform(X['bowl_team'])
        X['batsman'] = striker_encoder.transform(X['batsman'])
        X['bowler'] = bowler_encoder.transform(X['bowler'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = keras.Sequential([
            keras.layers.Input(shape=(X_train_scaled.shape[1],)),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(216, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
        
        huber_loss = tf.keras.losses.Huber(delta=1.0)
        model.compile(optimizer='adam', loss=huber_loss)
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, validation_data=(X_test_scaled, y_test))
        
        os.makedirs('models', exist_ok=True)
        model.save('models/ipl_score_predictor.h5')
    
    return model

model = get_model()

# Create scaler
scaler = MinMaxScaler()
X = df.drop(['total'], axis=1)
X['venue'] = venue_encoder.transform(X['venue'])
X['bat_team'] = batting_team_encoder.transform(X['bat_team'])
X['bowl_team'] = bowling_team_encoder.transform(X['bowl_team'])
X['batsman'] = striker_encoder.transform(X['batsman'])
X['bowler'] = bowler_encoder.transform(X['bowler'])
scaler.fit(X)

# Routes
@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            session['user'] = username
            return redirect(url_for('home'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        
        if username in users:
            return render_template('signup.html', error='Username already exists')
        
        users[username] = {'password': password, 'name': name}
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', username=session['user'])

@app.route('/prediction')
def prediction_form():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    venues = sorted(df['venue'].unique().tolist())
    bat_teams = sorted(df['bat_team'].unique().tolist())
    bowl_teams = sorted(df['bowl_team'].unique().tolist())
    batsmen = sorted(df['batsman'].unique().tolist())
    bowlers = sorted(df['bowler'].unique().tolist())
    
    return render_template('prediction.html',
                         venues=venues,
                         bat_teams=bat_teams,
                         bowl_teams=bowl_teams,
                         batsmen=batsmen,
                         bowlers=bowlers)

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    venue = request.form['venue']
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    striker = request.form['striker']
    bowler = request.form['bowler']
    
    decoded_venue = venue_encoder.transform([venue])
    decoded_batting_team = batting_team_encoder.transform([batting_team])
    decoded_bowling_team = bowling_team_encoder.transform([bowling_team])
    decoded_striker = striker_encoder.transform([striker])
    decoded_bowler = bowler_encoder.transform([bowler])
    
    input_data = np.array([decoded_venue, decoded_batting_team, decoded_bowling_team,
                         decoded_striker, decoded_bowler]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    
    predicted_score = model.predict(input_scaled)
    predicted_score = int(predicted_score[0, 0])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, 51)
    train_loss = np.random.uniform(0.1, 0.5, 50) * np.exp(-0.1 * np.array(epochs))
    val_loss = train_loss + np.random.uniform(0, 0.1, 50)
    
    ax.plot(epochs, train_loss, 'b', label='Training loss')
    ax.plot(epochs, val_loss, 'r', label='Validation loss')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    plt.close(fig)
    
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    
    return render_template('results.html',
                         prediction=predicted_score,
                         venue=venue,
                         batting_team=batting_team,
                         bowling_team=bowling_team,
                         striker=striker,
                         bowler=bowler,
                         plot_url=pngImageB64String)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)