# app/routes/predict_routes.py
import os

import numpy as np
from flask import Blueprint, request, jsonify
import mysql.connector
import tensorflow as tf
import pandas as pd
from google.cloud import storage
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict-lstm", methods=["POST"])
def predict():
    # Define the database connection parameters
    db_config = {
        'host': os.getenv('DB_HOST'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_DATABASE')
    }

    # Establish a connection to the database
    connection = mysql.connector.connect(**db_config)
    # Create a cursor object
    cursor = connection.cursor()

    if "user_id" in request.json:
        try:
            USER_ID = request.json["user_id"]

            # Define your SQL query with a placeholder for the user_id
            sql_query = "SELECT * FROM eat_logs WHERE user_id = %s"

            # Execute the SQL query with the user_id as a parameter
            cursor.execute(sql_query, (USER_ID,))
            result = cursor.fetchall()

            # Check if there are no results for the given user_id
            if not result:
                return jsonify({"error": "No data for the given user_id"}), 404

            # Create a DataFrame from the query result
            df = pd.DataFrame(result, columns=cursor.column_names)
            df['date'] = pd.to_datetime(df['date'])

            # Remove 'days' part from 'time' and convert to timedelta
            df['time'] = pd.to_timedelta(df['time'].astype(str).str.replace(' days ', ' ').str.replace(' day ', ' '))

            # Combine 'date' and 'time' columns to create 'Timestamp'
            df['Timestamp'] = df['date'] + df['time']

            # Initialize 'Pesan' column
            df['Pesan'] = None

            # Drop unnecessary columns if needed
            df = df[['Timestamp', 'calories', 'carbs', 'prots', 'fats']]

            # Resample the data by day and calculate the sum (or mean, etc.)
            daily_data = df.resample('D', on='Timestamp').sum()

            # Reset the index
            daily_data.reset_index(inplace=True)

            # Extract the time series columns
            series = daily_data[['calories', 'carbs', 'prots', 'fats']].values
            # Assuming 'series' contains your data with shape (n, 4)
            # where 'n' is the number of rows
            model = tf.keras.models.load_model('lstm_model - Final.h5')

            desired_sequence_length = 250
            number_of_features = 4

            # Pad or slice 'series' to achieve a length of 250 rows
            if len(series) < desired_sequence_length:
                # Pad the series with zeros to reach the desired length
                padded_data = np.zeros((desired_sequence_length, number_of_features))
                padded_data[-len(series):, :] = series  # Fill in the data at the end of the padded array
            else:
                # Take the last 250 rows if there are more than 250 rows
                padded_data = series[-desired_sequence_length:, :]

            # Reshape the padded data to match the expected shape
            reshaped_data = padded_data.reshape((1, desired_sequence_length, number_of_features))

            # Now, 'reshaped_data' has the shape (1, 250, 4)
            # You can use this reshaped data to make predictions using your trained model
            predicted_values = model.predict(reshaped_data)

            # Create a folder to store the images
            image_folder = "images"
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)

            # Plotting the predicted values
            plt.figure(figsize=(10, 6))
            features = ['calories', 'carbs', 'prots', 'fats']

            # Transpose the predicted values for better plotting
            predicted_values_transposed = list(map(list, zip(*predicted_values[0])))

            # Plot each feature
            for i, feature_values in enumerate(predicted_values_transposed):
                plt.plot(feature_values, label=f'{features[i]}')

            plt.xlabel('Time Step')
            plt.ylabel('Predicted Value')
            plt.legend()
            plt.title('Predicted Values Over Time for Each Feature')

            # Save the plot as an image
            first_plot_path = os.path.join(image_folder, 'predicted.png')
            plt.savefig(first_plot_path)
            plt.close()
            # Upload both images to Google Cloud Storage
            bucket_name = "sehatin-users-images"
            first_blob_name = f"{USER_ID}_predicted.png"
            first_blob = upload_to_storage(first_plot_path, bucket_name, first_blob_name)

            # Return the image URL in the JSON response
            return jsonify({"predicted_values_plot_url": first_blob.public_url})
        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({"error": "An error occurred"}), 500
        finally:
            # Close the database connection in the 'finally' block
            if connection.is_connected():
                cursor.close()
                connection.close()
    return jsonify({"error": "No data provided"}), 400


@predict_bp.route("/predict-pm", methods=["POST"])
def predictpm():
    # Define the database connection parameters
    db_config = {
        'host': os.getenv('DB_HOST'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_DATABASE')
    }

    # Establish a connection to the database
    connection = mysql.connector.connect(**db_config)
    # Create a cursor object
    cursor = connection.cursor()

    if "user_id" in request.json:
        try:
            USER_ID = request.json["user_id"]

            # Define your SQL query with a placeholder for the user_id
            sql_query = "SELECT * FROM eat_logs WHERE user_id = %s"

            # Execute the SQL query with the user_id as a parameter
            cursor.execute(sql_query, (USER_ID,))
            result = cursor.fetchall()

            # Check if there are no results for the given user_id
            if not result:
                return jsonify({"error": "No data for the given user_id"}), 404

            # Create a DataFrame from the query result
            df = pd.DataFrame(result, columns=cursor.column_names)
            df['date'] = pd.to_datetime(df['date'])

            # Remove 'days' part from 'time' and convert to timedelta
            df['time'] = pd.to_timedelta(df['time'].astype(str).str.replace(' days ', ' ').str.replace(' day ', ' '))

            # Combine 'date' and 'time' columns to create 'Timestamp'
            df['Timestamp'] = df['date'] + df['time']

            # Calculate the time difference between rows
            df['Selisih'] = df['Timestamp'].diff()

            # Convert time difference to hours
            df['Selisih'] = df['Selisih'].dt.total_seconds() / 3600

            # Initialize 'Pesan' column
            df['Pesan'] = None

            for i in range(1, len(df)):
                if df['calories'][i] >= 210 and df['calories'][i - 1] >= 210 and df['Selisih'][i] > 5 / 60:
                    if df['Selisih'][i] < 3:
                        df.at[i, 'Pesan'] = 'Rentang waktu makan kurang dari 3 jam'
                    elif df['Selisih'][i] > 5:
                        df.at[i, 'Pesan'] = 'Rentang waktu makan lebih dari 5 jam'
                else:
                    df.at[i, 'Pesan'] = "Rentang waktu makan yang baik"

            total = df[['calories', 'carbs', 'prots', 'fats']].sum()
            total_macro = total[['carbs', 'prots', 'fats']]
            # Define your SQL query with a placeholder for the user_id
            sql_query = "SELECT * FROM users_data WHERE user_id = %s"

            # Execute the SQL query with the user_id as a parameter
            cursor.execute(sql_query, (USER_ID,))
            result = cursor.fetchall()
            # Create a DataFrame from the query result
            df2 = pd.DataFrame(result, columns=cursor.column_names)
            recommended_calories = df2[['bmr']]
            # Pie chart for calories
            remaining_calories = max(0, recommended_calories['bmr'].values[0] - total['calories'])
            calories_sizes = [total['calories'], remaining_calories]

            # Create a folder to store the images
            image_folder = "images"
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)

            # Plotting the predicted values
            plt.figure(figsize=(10, 6))
            plt.pie(calories_sizes, labels=['Consumed', 'Remaining'], autopct='%1.1f%%', startangle=140)
            plt.title('Calories Consumption')
            plt.axis('equal')

            # Save the first plot as an image
            first_plot_path = os.path.join(image_folder, 'calories_consumption_plot.png')
            plt.savefig(first_plot_path)
            plt.close()

            # Create a pie chart for macronutrients proportion
            plt.figure(figsize=(6, 6))
            plt.pie(total_macro, labels=total_macro.index, autopct='%1.1f%%')
            plt.title('Proportion of Carbs, Prots, and Fats')

            second_plot_path = os.path.join(image_folder, 'macronutrient_proportion_plot.png')
            plt.savefig(second_plot_path)
            plt.close()

            # Upload both images to Google Cloud Storage
            bucket_name = "sehatin-users-images"

            # Upload the first plot
            first_blob_name = f"{USER_ID}_calories_consumption_plot.png"
            first_blob = upload_to_storage(first_plot_path, bucket_name, first_blob_name)

            # Upload the second plot
            second_blob_name = f"{USER_ID}_macronutrient_proportion_plot.png"
            second_blob = upload_to_storage(second_plot_path, bucket_name, second_blob_name)

            # Calculate the total amount of carbs, proteins, and fats over all days
            total_carbs = df['carbs'].sum()
            total_prots = df['prots'].sum()
            total_fats = df['fats'].sum()

            # Calculate the total amount of these three macronutrients
            total_all = total_carbs + total_prots + total_fats

            # Calculate the proportion of each macronutrient
            carbs_prop = total_carbs / total_all
            prots_prop = total_prots / total_all
            fats_prop = total_fats / total_all

            # Check if the proportion of each macronutrient is within the AMDR
            message = ""
            if not 0.45 <= carbs_prop <= 0.65:
                message = 'Warning: The proportion of carbs is out of the acceptable range (45–65%). '
            if not 0.20 <= fats_prop <= 0.35:
                message = 'Warning: The proportion of fats is out of the acceptable range (20–35%). '
            if not 0.10 <= prots_prop <= 0.35:
                message = 'Warning: The proportion of protein is out of the acceptable range (10–35%). '
            if message == "":
                message = 'You have a good proportion of carbohydrate, protein, and fat.'

            # Return the URLs of the uploaded images and the message in the JSON response
            return jsonify({
                "calories_consumption_plot_url": first_blob.public_url,
                "macronutrient_proportion_plot_url": second_blob.public_url,
                "message": message
            })
        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({"error": "An error occurred"}), 500
        finally:
            # Close the database connection in the 'finally' block
            if connection.is_connected():
                cursor.close()
                connection.close()
    return jsonify({"error": "No data provided"}), 400


def upload_to_storage(file_path, bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_path)
    # Delete the local file after upload
    if os.path.exists(file_path):
        os.remove(file_path)
    return blob
