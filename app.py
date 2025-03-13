from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle

# Load the trained Decision Tree Regressor model
dtr = pickle.load(open('dtr.pkl', 'rb'))

# Load the preprocessor (ColumnTransformer with StandardScaler and OneHotEncoder)
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Load the merged dataset
merged_df = pd.read_csv('c:/Users/User/Desktop/final year model/merged_dataset.csv')

# Initialize the Flask app
app = Flask(__name__)

# Define the route for making predictions
@app.route("/api/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input features from the JSON request
        data = request.get_json()
        Year = int(data['Year'])
        Item = data['Item']

        # Filter the merged dataset to get the average values for the given year and item
        filtered_df = merged_df[(merged_df['Year'] == Year) & (merged_df['Item'] == Item)]
        
        if filtered_df.empty:
            return jsonify({"error": "No data available for the given year and item."}), 404

        # Calculate the mean values for the other features
        average_rain_fall_mm_per_year = filtered_df['average_rain_fall_mm_per_year'].mean()
        pesticides_tonnes = filtered_df['pesticides_tonnes'].mean()
        avg_temp = filtered_df['avg_temp'].mean()
        Area = filtered_df['Area'].mode()[0]  # Use the most frequent area

        # Create a numpy array with the input features
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)

        # Transform the features using the preprocessor
        transformed_features = preprocessor.transform(features)

        # Make a prediction using the trained model
        prediction = dtr.predict(transformed_features).reshape(1, -1)

        # Return the prediction as a JSON response
        return jsonify({"prediction": prediction[0][0]})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)