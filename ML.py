import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")
from tabulate import tabulate
import tkinter as tk
from tkinter import filedialog
import os
# Set pandas display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#function to load the dataset
def load_data():
    """Load your dataset from file (csv, json, excel, html)"""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    filepath = filedialog.askopenfilename(title="Select your dataset file")
    _, file_extension = os.path.splitext(filepath)
    if file_extension == '.csv':
        data = pd.read_csv(filepath)
    elif file_extension == '.json':
        df = pd.read_json(filepath)
    elif file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath)
    elif file_extension == '.html':
        df = pd.read_html(filepath)[0]
    else:
        raise ValueError("Unsupported file format. Please upload a CSV, JSON, Excel, or HTML file.")

    return data

#function to split the dataset into train and test sets
def split_data(data, target_col):
    """Split the dataset into train and test sets."""
    X = data.drop(columns=[target_col])
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

#Encoding categorical variables
def encode_categorical(data):
    """Encode categorical variables using Label Encoding or One-Hot Encoding."""
    while True:
        print("Select the column you want to encode:")
        print(data.select_dtypes(include=['object']).columns) # Display the categorical columns in the dataset
        col=input("Enter the column name (or type exit to stop): ")
        if col.lower()=='exit':
            print("Encoding process stopped")
            break
        print("Select the encoding method:")
        print("1. Label Encoding (for ordinal data)")
        print("2. One-Hot Encoding (for nominal data)")
        print("3. Ordinal Encoding (for ordinal data)")
        print("4. Binary Encoding (for high cardinality data)")
        print("5. Target (Mean) Encoding (for categorical data)")
        print("6. Frequency Encoding (for high cardinality data)")
        print("7.Hash Encoding (for high cardinality data)")
        select_method = input("Enter the number of the method you want to use:")
        if select_method == '1': #Label Encoding
            for col in data.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
        elif select_method == '2': # One-Hot Encoding
            data = pd.get_dummies(data, columns=[col], drop_first=True)
        elif select_method=='3': #Ordinal Encoding
            from sklearn.preprocessing import OrdinalEncoder
            oe = OrdinalEncoder()
            data[col] = oe.fit_transform(data[[col]])
        elif select_method=='4': #Binary Encoding
            from category_encoders import BinaryEncoder
            be = BinaryEncoder()
            data = be.fit_transform(data[col])
            data.drop(col, axis=1, inplace=True)
        elif select_method=='5':
            from category_encoders import TargetEncoder
            te = TargetEncoder()
            data[col] = te.fit_transform(data[col], data['target'])
            data.drop(col, axis=1, inplace=True)
        elif select_method=='6':
            from category_encoders import FrequencyEncoder
            fe = FrequencyEncoder()
            data = fe.fit_transform(data[col])
            data.drop(col, axis=1, inplace=True)
        elif select_method=='7':
            from category_encoders import HashingEncoder
            he = HashingEncoder()
            data = he.fit_transform(data[col])
            data.drop(col, axis=1, inplace=True)
        else:
            print("Invalid method selected. Please select a valid method (1 OR 2).")
    return data

#function to scale the dataset
def scale_data(X_train, X_test):
    """Scale the dataset using StandardScaler or MinMaxScaler."""
    print("Select the scaling method:")
    print("1. StandardScaler")
    print("2. MinMaxScaler")
    select_method = input("Enter the number of the method you want to use: ")
    if select_method == '1':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    elif select_method == '2':
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        print("Invalid method selected. Defaulting to StandardScaler.")
        scaler = StandardScaler() # As a default
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled data back to DataFrame with original column names
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    return X_train, X_test, scaler

# Model training and evaluation function
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """Train and evaluate a machine learning model."""
    print("Select the model you want to use:")
    print("1. Linear Regression")
    print("2. SGD Regressor")
    print("3. Decision Tree Regressor")
    print("4. Random Forest Regressor")
    select_model = input("Enter the number of the model you want to use: ")
    if select_model == '1':
        model = LinearRegression()
    elif select_model == '2':
        model = SGDRegressor()
    elif select_model == '3':
        model = DecisionTreeRegressor()
    elif select_model == '4':
        model = RandomForestRegressor()
    else:
        print("Invalid model selected. Defaulting to Linear Regression.")
        model = LinearRegression()
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    #print(f"y_pred:{y_pred}")
    # Evaluate the model
    y_train_pred=model.predict(X_train)
    #print(f"y_train_pred:{y_train_pred}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test,y_pred)}")
    print(f"Mean Squared Error: {mean_squared_error(y_test,y_pred)}")
    print(f"R2 Score: {r2_score(y_test,y_pred)}")
    return model

#Predict on new data
def predict_on_new_data(model, scaler, X_train):
    """Predict on new data."""
    while True:
        cont_pred = input("Do you want to continue with the prediction? (yes/no): ")
        if cont_pred.lower() == 'no':
            print("Prediction process stopped")
            break
        elif cont_pred.lower() == 'yes':
            print("Select the new data file:")
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            filepath = filedialog.askopenfilename(title="Select your new data file")
            _, file_extension = os.path.splitext(filepath)
            if file_extension == '.csv':
                new_data = pd.read_csv(filepath)
            elif file_extension == '.json':
                new_data = pd.read_json(filepath)
            elif file_extension in ['.xls', '.xlsx']:
                new_data = pd.read_excel(filepath)
            elif file_extension == '.html':
                new_data = pd.read_html(filepath)[0]
            else:
                raise ValueError("Unsupported file format. Please upload a CSV, JSON, Excel, or HTML file.")
            
            # Ensure new_data has the same columns as X_train
            print("Reindexing new data to match training data columns...")
            new_data = new_data.reindex(columns=X_train.columns, fill_value=0)

            # Encode categorical variables in the new data
            #print("Encoding categorical variables in the new data...")
            #new_data = encode_categorical(new_data)

            # Scale the new data using the same scaler as the training data
            print("Scaling the new data...")
            new_data = scaler.transform(new_data)

            # Make predictions on the new data
            print("Making predictions on the new data...")
            predictions = model.predict(new_data)
            print("Predictions on new data:")
            #print(predictions)

            # Save predictions to a CSV file
            output_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            pd.DataFrame(predictions, columns=["Predictions"]).to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}.")

#Main function to run the ML pipeline
def main():
    # Load the dataset
    data = load_data()
    print("Dataset loaded successfully.")
    
    # Encoding categorical variables
    print("Encoding categorical variables...")
    data = encode_categorical(data)
    print("Categorical variables encoded successfully.")

    # Split the dataset into train and test sets
    print(data.columns)
    target_col = input("Enter the target column name: ")
    X_train, X_test, y_train, y_test = split_data(data, target_col)
    print("Dataset split into train and test sets.")
    #Scale the dataset
    print("Scaling the dataset...")
    X_train, X_test,scalar = scale_data(X_train, X_test)
    print("Dataset scaled successfully.")
    # Train and evaluate the model
    print("Training and evaluating the model...")
    model=train_and_evaluate_model(X_train, y_train, X_test, y_test)
    #model=train_and_evaluate_model(X_train, y_train, X_test, y_test)
    print("Model training and evaluation completed.")  

    # Predict on new data
    print("Predicting on new data...")
    
    predict_on_new_data(model, scalar, X_train)
    print("Prediction completed.")
     
main()
