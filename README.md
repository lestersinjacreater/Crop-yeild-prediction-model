# Crop Yield Prediction

The project "Crop Yield Prediction" aims to predict the yield of various crops based on several features such as year, average rainfall, pesticide usage, average temperature, area, and crop type.

## Data

The dataset used in this project contains information about crop yields from different countries over several years. The key features in the dataset include:
- **Year**: The year of the data record.
- **average_rain_fall_mm_per_year**: The average annual rainfall in millimeters.
- **pesticides_tonnes**: The amount of pesticides used in tonnes.
- **avg_temp**: The average temperature in degrees Celsius.
- **Area**: The country or region where the data was recorded.
- **Item**: The type of crop.
- **hg/ha_yield**: The yield of the crop in hectograms per hectare.

## Data Preprocessing

The data preprocessing steps include:
1. **Handling Missing Values**: Checking for and handling any missing values in the dataset.
2. **Removing Duplicates**: Identifying and removing duplicate records.
3. **Encoding Categorical Variables**: Using `OneHotEncoder` to convert categorical variables (Area and Item) into numerical format.
4. **Scaling Numerical Features**: Using `StandardScaler` to standardize numerical features (Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp).

## Feature Engineering

The features are divided into numerical and categorical features:
- **Numerical Features**: Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp.
- **Categorical Features**: Area, Item.

A `ColumnTransformer` is used to apply the `StandardScaler` to numerical features and `OneHotEncoder` to categorical features.

## Model Training

Several regression models are trained to predict crop yield:
1. **Linear Regression**
2. **Lasso Regression**
3. **Ridge Regression**
4. **Decision Tree Regressor**
5. **K-Nearest Neighbors (KNN) Regressor**

## Model Evaluation

The models are evaluated using the following metrics:
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between the predicted and actual values.
- **R² Score**: Indicates how well the model explains the variance in the data (with 1 being a perfect fit).

## Results

The results show the performance of each model:
- **Linear Regression**: MAE: 29897.28, R²: 0.747
- **Lasso Regression**: MAE: 29883.83, R²: 0.747 (with a convergence warning)
- **Ridge Regression**: MAE: 29852.96, R²: 0.747
- **Decision Tree Regressor**: MAE: 5802.63, R²: 0.963
- **KNN Regressor**: MAE: 4693.99, R²: 0.985

The Decision Tree and KNN models perform significantly better than the linear models.

## Predictive System

A function `prediction` is defined to predict the crop yield based on input features. The function uses the trained `DecisionTreeRegressor` model and the preprocessor to transform the input features before making predictions.

## Saving the Model

The trained model and preprocessor are saved using `pickle` for future use.

## Web Interface

A simple web interface is created using Flask to allow users to input features and get crop yield predictions. The interface includes fields for Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, and Item.

## Conclusion

The project successfully demonstrates the use of various regression models to predict crop yields based on historical data. The Decision Tree and KNN models show the best performance, indicating their suitability for this task. The project also includes a web interface for easy user interaction.


## How to Run the Project

To run the project after cloning the repository from GitHub, follow these steps:

1. **Clone the Repository**:
   ```sh
   git clone <https://github.com/lestersinjacreater/Crop-yeild-prediction-model.git>
   cd <repository-directory>

2. **Set Up the Environment**:
     Create a virtual environment (optional but recommended):
     ```sh
    python -m venv venv

Activate the virtual environment:
        **On Windows and macOS**:
        ```sh
        windows-venv\Scripts\activate
        macOS-source venv/bin/activate



3. **Install Dependencies:**:
    ```sh 
    pip install -r requirements.txt
    
4.    **Prepare the Dataset**: Ensure that your dataset (yield_df.csv) is in the correct location as expected by your code.
        
5. **Run the Flask Application**:
   ```sh 
   python app.py
6.**Access the Application**:
 Open your web browser and go to http://127.0.0.1:5000/ to access the application.

7.**Use the Web Interface**:

Enter the required features such as Year, Average Rainfall, Pesticides, Average Temperature, Area, and Item in the form provided on the web page.
Click the "Predict" button to get the predicted crop yield.   


