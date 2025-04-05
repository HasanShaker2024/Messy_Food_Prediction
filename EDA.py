import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
from tabulate import tabulate
import tkinter as tk
from tkinter import filedialog
import os
# Set pandas display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Function to load the dataset
def load_dataset():
    """Load your dataset from file (csv, json, excel, html)"""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    filepath = filedialog.askopenfilename(title="Select your dataset file")
    _, file_extension = os.path.splitext(filepath)
    if file_extension == '.csv':
        df = pd.read_csv(filepath)
    elif file_extension == '.json':
        df = pd.read_json(filepath)
    elif file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath)
    elif file_extension == '.html':
        df = pd.read_html(filepath)[0]
    else:
        raise ValueError("Unsupported file type. Please select a csv, json, excel, or html file.")
    return df, filepath

# Function to display basic information of the dataset
def basic_info(data):
    """ Display Basic Information of Dataset"""
    cols_name, counts, data_type, nulls, duplicates, uniques, means, std_dev, minimum, quantile_25, quantile_50, quantile_75, maximum, top, frequency, distribution_type = [],[], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for col in data.columns:
        cols_name.append(col)
        counts.append(data[col].count())
        data_type.append(data[col].dtype)
        nulls.append(data[col].isnull().sum())
        duplicates.append(data.duplicated().sum())
        uniques.append(data[col].nunique())
        
        if pd.api.types.is_numeric_dtype(data[col]):
            means.append(data[col].mean())
            std_dev.append(data[col].std())
            minimum.append(data[col].min())
            quantile_25.append(data[col].quantile(0.25))
            quantile_50.append(data[col].quantile(0.5))
            quantile_75.append(data[col].quantile(0.75))
            maximum.append(data[col].max())
            top.append("N/A")
            frequency.append("N/A")
            skewness=data[col].skew() #Check the skewness of the data
            if skewness<0.5:
                distribution_type.append("Normal")
            elif skewness>0.5:
                distribution_type.append("Right Skewed")
            else:
                distribution_type.append("Left Skewed")
        else:
            means.append("N/A")
            std_dev.append("N/A")
            minimum.append("N/A")
            quantile_25.append("N/A")
            quantile_50.append("N/A")
            quantile_75.append("N/A")
            maximum.append("N/A")
            top.append(data[col].mode()[0])
            frequency.append(data[col].value_counts().max())
            distribution_type.append("N/A")
    
    data_info = pd.DataFrame({
        "Column": cols_name,
        "Counts": counts,
        "Data Type": data_type,
        "No of Nulls": nulls,
        "No of Duplicates": duplicates,
        "No of Uniques": uniques,
        "Mean": means,
        "Std Dev": std_dev,
        "Minimum": minimum,
        "25%": quantile_25,
        "50%": quantile_50,
        "75%": quantile_75,
        "Maximum": maximum,
        "Top": top,
        "Frequency": frequency,
        "distribution_type": distribution_type
    })
    return data_info

# Function to remove duplicates
def remove_duplicates(data):
    """ Remove duplicate rows from the dataset."""
    data.drop_duplicates(inplace=True)
    return data

# Function to remove unwanted columns    
def remove_unwanted_cols(data,filepath):
    print("Select the columns you want to remove (or type exit to stop removing columns):")
    print(data.columns)
    while True:
        col = input("Enter the column name: ").strip() # Remove leading and trailing whitespaces
        if col.lower() == 'exit':
            print("Column removal stopped")
            break
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)
            print(f"{col} removed successfully")
        else:
            print("Invalid column name. Please select a valid column.")
    return data

#Change the data type of the columns
def change_data_type(data):
    print("Select the column you want to chnge the data type:")
    print(data.columns)
    while True:
        col=input("Enter the column name (or type exit to stop): ")
        if col.lower()=='exit':
            print("Data type change stopped")
            break
        if col in data.columns:
            print("Select the data type you want to change to:")
            print("1. Numeric")
            print("2. Categorical")
            select_type=input("Enter the number of the data type you want to change to: ")
            if select_type=='1':
                data[col]=pd.to_numeric(data[col],errors='coerce')
            elif select_type=='2':
                data[col]=data[col].astype('category')
            else:
                print("Invalid data type selected. Please select a valid data type (1 OR 2).")
    return data
            
# Function to manage missing values
def manage_nulls(data):
    """Manage missing values in the dataset."""
    print("Select the column you want to handle its Null:")
    print(data.columns)
    while True:
        col=input("Enter the column name (or type exit to stop): ")
        if col.lower()=='exit':
            print("Null handling stopped")
            break
        
        if col in data.columns:
            print("Select the method to manage missing values:")
            print("1. Drop missing values")
            print("2. Fill missing values with mean/mode")
            print("3. Fill missing values using interpolation")
            print("4. Fill missing values using transform/group by method")
            select_method = input("Enter the number of the method you want to use: ")
            if select_method == '1':
                print("Dropping missing values...")
                data.dropna(inplace=True)
            elif select_method == '2':
                print("Filling missing values with mean/mode...")
                for col in data.columns:
                    if data[col].isnull().sum() > 0:
                        if pd.api.types.is_numeric_dtype(data[col]):
                            data[col].fillna(data[col].mean(), inplace=True)
                        else:
                            data[col].fillna(data[col].mode()[0], inplace=True)
            elif select_method == '3':
                print("Filling missing values using interpolation...")
                for col in data.columns:
                    if data[col].isnull().sum() > 0:
                        if pd.api.types.is_numeric_dtype(data[col]):
                            data[col].interpolate(method='linear', inplace=True)
                        else:
                            print("Categorical data cannot be interpolated.")
            elif select_method == '4':
                print("Filling missing values using transform/group by method...")
                for col in data.columns:
                    if data[col].isnull().sum() > 0:
                        if pd.api.types.is_numeric_dtype(data[col]):
                            col_filter= input("Enter the column name to group by: ")
                            data[col].fillna(data.groupby(col_filter).transform('mean'), inplace=True)
                        else:
                            print("Categorical data cannot be filled using transform/group by method.")

            else:
                print("Invalid method selected. Please select a valid method (1 OR 2).")
    return data
# Function to handle outliers
def handle_outliers(data,col, method='iqr'):
    """Detect and handle outliers using the specified method."""
    if pd.api.types.is_numeric_dtype(data[col]):
        print("Select the method to handle outliers:")
        print("1. IQR (for skewed data)")
        print("2. Z-score (for normal data)")
        select_method=input("Enter the number of the method you want to use: ")
        if select_method == '1':
            method = 'iqr'
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR
            data = data[(data[col] >= lower_limit) & (data[col] <= upper_limit)]
        elif select_method == '2':
            method = 'zscore'
            from scipy.stats import zscore
            data = data[(zscore(data[col]) < 3)]
        else:
            print("Invalid method selected. Please select a valid method (1 OR 2).")
    else:
        print("The column is not numeric. Please select a numeric column to handle outliers.")
    return data

# Function to save the output to a CSV file
def save_output(data,filepath):
    """Save the basic informtion data frame as an output to a CSV file."""
    # Extract the original file name and create the output file name
    original_file_name = os.path.splitext(os.path.basename(filepath))[0]
    directory = os.path.dirname(filepath)
    output_file_name = f"{original_file_name}_Dataset_Info.csv"
    output_file_path = os.path.join(directory, output_file_name)
    data.to_csv(output_file_path, index=True)
    print(f"Output file has been saved to {output_file_path}")

# Function to save the updated dataset
def save_modified_dataset(data,filepath):
      # Save the updated DataFrame back to the original file
    save_option = input("Do you want to save the updated dataset? (yes/no): ").strip().lower()
    if save_option == 'yes':
        directory = os.path.dirname(filepath)  # Extract the directory
        original_file_name = os.path.basename(filepath)  # Extract the file name
        modified_file_name = f"Modified_{original_file_name}"  # Create the modified file name
        modified_file_path = os.path.join(directory, modified_file_name)  # Combine directory and modified file name
        if filepath.endswith('.csv'):
            data.to_csv(modified_file_path, index=False)
        elif filepath.endswith('.json'):
            data.to_json(modified_file_path, index=False)
        elif filepath.endswith('.xls') or filepath.endswith('.xlsx'):
            data.to_excel(modified_file_path, index=False)
        elif filepath.endswith('.html'):
            data.to_html(modified_file_path, index=False)
        print(f"Updated dataset saved to {modified_file_path}")
    return filepath

# Main function to call all the functions
def main():
    df, filepath = load_dataset()   # Load the dataset
    info = basic_info(df) # Get the basic information of the dataset
    print(tabulate(info, headers='keys', tablefmt='psql')) # usin tabulate to organize the output table
    save_output(info,filepath) # Save the output to a CSV file

    # Data Preprocessing

    #Remove unwanted columns
    print("Unwanted Columns Removal")
    df=remove_unwanted_cols(df,filepath) # Remove unwanted columns from the dataset

    # Handling Duplicates
    print("Duplicates Handling") 
    df=remove_duplicates(df) # Remove duplicates from the dataset
    print("Duplicates removed successfully")

    # Handling Outliers
    print("Outliers Handling")
    print("Select the column you want to handle outliers:")
    print(df.select_dtypes(include=["number"]).columns) # Display the numeric columns in the dataset
    while True:
        col = input("Enter the column name (or type exit to stop outliers handling): ")
        if col.lower() == 'exit':
            print("Outliers handling stopped")
            break
        if col in df.columns:
            
            df = handle_outliers(df,col) # Handle outliers in the selected column
            print("Outliers handled successfully")
            df.reset_index(drop=True, inplace=True) # Reset the index after removing rows
        else:
            print("Invalid column name. Please select a valid column.")
    save_modified_dataset(df,filepath) # Save the updated dataset   
main()
