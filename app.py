from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn

# Print the version of scikit-learn
print(sklearn.__version__)

# Load the trained Decision Tree Regressor model
dtr = pickle.load(open('dtr.pkl', 'rb'))

# Load the preprocessor (ColumnTransformer with StandardScaler and OneHotEncoder)
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Initialize the Flask app
app = Flask(__name__)

# Define the route for the home page
@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')

# Define the route for making predictions
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input features from the form
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']

        # Create a numpy array with the input features
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)

        # Transform the features using the preprocessor
        transformed_features = preprocessor.transform(features)

        # Make a prediction using the trained model
        prediction = dtr.predict(transformed_features).reshape(1, -1)

        # Render the index.html template with the prediction result
        return render_template('index.html', prediction=prediction[0][0])

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)