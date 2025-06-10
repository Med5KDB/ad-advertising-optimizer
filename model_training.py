import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

data = load_data('bakery_advertising_data.csv')

def preprocess_data(df, training=True, feature_names=None):
    """Preprocess the dataset for linear regression."""
    df_processed = pd.get_dummies(df, columns=['Canal', 'Mois', 'Jour_Semaine'])
    
    if training:
        X = df_processed.drop(columns=['Date', 'Chiffre_Affaires', 'ROI'])
        y = df_processed['Chiffre_Affaires']
        return X, y, X.columns
    else:
        for col in feature_names:
            if col not in df_processed.columns:
                df_processed[col] = 0
        return df_processed[feature_names]

X, y, feature_names = preprocess_data(data, training=True)

def train_model(X, y):
    """Train a linear regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test
model, X_test, y_test = train_model(X, y)

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
evaluate_model(model, X_test, y_test)

def predict_revenue(model, new_data, feature_names):
    """Predict revenue using the trained model."""
    new_data_processed = preprocess_data(new_data, training=False, feature_names=feature_names)
    predictions = model.predict(new_data_processed)
    return predictions

new_data = pd.DataFrame({
    'Budget': [1000, 1500],
    'Impressions': [50000, 75000],
    'Clics': [500, 750],
    'Conversions': [50, 75],
    'CPC': [2.0, 2.0],
    'CPA': [20.0, 20.0],
    'Facteur_Saisonnier': [1.0, 1.0],
    'Budget_Total_Semaine': [10000, 15000],
    'Canal': ['Facebook', 'Google'], 
    'Mois': ['Janvier', 'Février'],  
    'Jour_Semaine': ['Lundi', 'Mardi'] 
})
predictions = predict_revenue(model, new_data, feature_names)
print("Predicted Revenue for New Data:")
print(predictions)