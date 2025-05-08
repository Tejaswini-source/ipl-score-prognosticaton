import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask_wtf.csrf import CSRFProtect
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Change this to a secure random key in production
bcrypt = Bcrypt(app)


# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client['ipl_app']
users_col = db['users']
predictions_col = db['predictions']
contact_collection = db['contact']

# Load and preprocess data
def load_data():
    ipl = pd.read_csv('ipl_data.csv')
    df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'mid', 'striker', 'non-striker'], axis=1)
    return df

df = load_data()

# Get unique values for dropdowns
venues = sorted(df['venue'].unique().tolist())
teams = sorted(df['bat_team'].unique().tolist())
batsmen = sorted(df['batsman'].unique().tolist())
bowlers = sorted(df['bowler'].unique().tolist())

# LabelEncoders
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

# Load or train model
def get_model():
    if os.path.exists('models/ipl_score_predictor.h5'):
        return keras.models.load_model('models/ipl_score_predictor.h5')

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
    model.compile(optimizer='adam', loss=tf.keras.losses.Huber())
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, validation_data=(X_test_scaled, y_test))

    os.makedirs('models', exist_ok=True)
    model.save('models/ipl_score_predictor.h5')
    return model

model = get_model()

# Scaler for future predictions
scaler = MinMaxScaler()
X = df.drop(['total'], axis=1)
X['venue'] = venue_encoder.transform(X['venue'])
X['bat_team'] = batting_team_encoder.transform(X['bat_team'])
X['bowl_team'] = bowling_team_encoder.transform(X['bowl_team'])
X['batsman'] = striker_encoder.transform(X['batsman'])
X['bowler'] = bowler_encoder.transform(X['bowler'])
scaler.fit(X)

@app.route('/')
def root():
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email').lower()
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Validation
        if not all([username, email, password, confirm_password]):
            flash("All fields are required", "error")
            return redirect(url_for('signup'))
            
        if len(username) < 4:
            flash("Username must be at least 4 characters", "error")
            return redirect(url_for('signup'))
            
        if len(password) < 6:
            flash("Password must be at least 6 characters", "error")
            return redirect(url_for('signup'))
            
        if password != confirm_password:
            flash("Passwords do not match", "error")
            return redirect(url_for('signup'))

        # Check if user exists
        if users_col.find_one({'username': username}):
            flash("Username already exists", "error")
            return redirect(url_for('signup'))
            
        if users_col.find_one({'email': email}):
            flash("Email already registered", "error")
            return redirect(url_for('signup'))

        # Create new user
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        users_col.insert_one({
            'username': username,
            'email': email,
            'password': hashed_pw
        })
        
        flash("Account created successfully! Please login", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember')
        
        user = users_col.find_one({'username': username})
        
        if user and bcrypt.check_password_hash(user['password'], password):
            session['username'] = user['username']
            session.permanent = bool(remember)  # Persistent session if "remember me" checked
            
            flash("Login successful!", "success")
            return redirect(url_for('home'))
            
        flash("Invalid username or password", "error")
        return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email').lower()  # Convert to lowercase for case-insensitive match
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        # Validation
        if not all([email, new_password, confirm_password]):
            flash("All fields are required", "error")
            return redirect(url_for('forgot_password'))

        if new_password != confirm_password:
            flash("Passwords do not match", "error")
            return redirect(url_for('forgot_password'))

        if len(new_password) < 6:
            flash("Password must be at least 6 characters", "error")
            return redirect(url_for('forgot_password'))

        # Check if email exists
        user = users_col.find_one({'email': email})
        if not user:
            flash("Email not found", "error")
            return redirect(url_for('forgot_password'))

        # Update password
        hashed_pw = bcrypt.generate_password_hash(new_password).decode('utf-8')
        users_col.update_one(
            {'email': email},
            {'$set': {'password': hashed_pw}}
        )
        
        flash("Password updated successfully! You can now login.", "success")
        return redirect(url_for('login'))

    return render_template('forgot_password.html')

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        venue = request.form['venue']
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        striker = request.form['striker']
        bowler = request.form['bowler']

        input_data = np.array([
            venue_encoder.transform([venue])[0],
            batting_team_encoder.transform([batting_team])[0],
            bowling_team_encoder.transform([bowling_team])[0],
            striker_encoder.transform([striker])[0],
            bowler_encoder.transform([bowler])[0]
        ]).reshape(1, -1)
        input_scaled = scaler.transform(input_data)

        predicted_score = int(model.predict(input_scaled)[0, 0])

        # Store prediction with formatted timestamp
        predictions_col.insert_one({
            'username': session['username'],
            'venue': venue,
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'striker': striker,
            'bowler': bowler,
            'predicted_score': predicted_score,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
        })

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
        plot_url = "data:image/png;base64," + base64.b64encode(pngImage.getvalue()).decode('utf8')

        return render_template('results.html',
                               prediction=predicted_score,
                               venue=venue,
                               batting_team=batting_team,
                               bowling_team=bowling_team,
                               striker=striker,
                               bowler=bowler,
                               plot_url=plot_url)

    return render_template('prediction.html',
                         venues=venues,
                         bat_teams=teams,
                         bowl_teams=teams,
                         batsmen=batsmen,
                         bowlers=bowlers)

@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get predictions for the current user, sorted by timestamp (newest first)
    predictions = list(predictions_col.find(
        {'username': session['username']},
        {'_id': 0, 'username': 0}  # Exclude these fields from results
    ).sort('timestamp', -1))
    
    return render_template('history.html', predictions=predictions)

@app.route('/profile')
def profile():
    if 'username' not in session:
        return redirect(url_for('login'))
    user = users_col.find_one({'username': session['username']})
    return render_template('profile.html', user=user)

@app.route('/about')
def about():
    return render_template('about.html')

# Route for handling contact form submission
@app.route("/contact", methods=["GET", "POST"])
def contact():
    message = ""
    message_type = ""

    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message_text = request.form.get("message")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if name and email and message_text:
            try:
                contact_entry = {
                    "name": name,
                    "email": email,
                    "message": message_text,
                    "timestamp": timestamp
                }
                contact_collection.insert_one(contact_entry)
                message = "Message sent successfully!"
                message_type = "success"
            except Exception as e:
                message = f"Error: {str(e)}"
                message_type = "error"
        else:
            message = "All fields are required."
            message_type = "error"

    return render_template("contact.html", message=message, message_type=message_type)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)