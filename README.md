# Medical-Recommendation-System

Overview
The Medical Recommendation System is a web application designed to help users diagnose potential diseases based on their symptoms and provide tailored recommendations, including treatments, precautions, diets, and workout routines. The system uses machine learning models (SVM, Random Forest, and KNN) for disease prediction and displays an interactive user interface built using Streamlit.

Features
Symptom-based Diagnosis: Users can input their symptoms, and the system predicts the most likely disease using an SVM model.
Recommendations: Provides detailed recommendations for medications, precautions, diets, and workout routines based on the predicted disease.
Interactive Interface: A user-friendly interface built with Streamlit allows users to select symptoms and receive instant results.
Technologies Used
Programming Language: Python
Libraries:
pandas
Streamlit
scikit-learn
pickle
numpy
Machine Learning Models:
Support Vector Machine (SVM)
Random Forest
K-Nearest Neighbors (KNN)
File Structure
Medical_Recommendation_System.py: Main Python script containing the entire implementation.
training_data.csv: Dataset used for training the machine learning models.
medications.csv, precautions_df.csv, diets.csv, description.csv, workout_df.csv: Supplementary datasets for recommendations related to medications, precautions, diets, and workouts.
svm.pkl: Serialized SVM model used for disease prediction.
Stethescope.jpg: Image used in the Streamlit UI.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/medical-recommendation-system.git
Navigate to the project directory:

bash
Copy code
cd medical-recommendation-system
Install the required Python packages:

bash
Copy code
pip install pandas streamlit scikit-learn
Make sure the required datasets (training_data.csv, medications.csv, precautions_df.csv, diets.csv, description.csv, workout_df.csv) are located in the same directory as the script.

Usage
Train the model (if not already trained and serialized):

The script automatically trains the models (SVM, Random Forest, KNN) and saves the SVM model as svm.pkl.
Run the Streamlit application:

bash
Copy code
streamlit run Medical_Recommendation_System.py
Open your web browser and go to the local Streamlit URL (usually http://localhost:8501).

Enter your name and select your symptoms from the multi-select dropdown.

Click the "Diagnose" button to receive a diagnosis and tailored recommendations.

Machine Learning Workflow
Data Preprocessing:

LabelEncoder is applied to the prognosis column in the training dataset to convert disease labels into numerical values.
Model Training:

The dataset is split into training and testing sets (x_train, x_test, y_train, y_test) with a 75-25 split.
Three models (SVM, Random Forest, KNN) are trained on the training set.
Model Testing:

The models are tested on the test set, and accuracy scores are calculated.
SVM Model Serialization:

The trained SVM model is saved as a pickle file (svm.pkl) for future predictions.
Disease Prediction:

The user's selected symptoms are mapped to a vector using a predefined dictionary (symptoms_dict), which is passed to the SVM model for prediction.
Recommendations:

Based on the predicted disease, the application fetches recommendations (description, medications, precautions, diets, and workout) from the corresponding datasets.
Future Enhancements
Integrate additional machine learning models for better accuracy.
Add more diseases and symptoms to the dataset.
Enhance the user interface for a more dynamic and user-friendly experience.
Add a feedback loop for users to confirm diagnosis accuracy and recommendations.
License
This project is licensed under the MIT License.
