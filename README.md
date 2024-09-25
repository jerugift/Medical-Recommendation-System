# Medical Recommendation System

### README

#### Introduction

The **Medical Recommendation System** is a Python-based web application built using **Streamlit** that predicts diseases based on symptoms provided by the user and recommends appropriate treatments, precautions, diets, and workout routines. The system employs multiple machine learning models like **Support Vector Machine (SVM)**, **Random Forest**, and **K-Nearest Neighbors (KNN)** to predict diseases based on symptoms and provides personalized medical recommendations.

#### Features
1. **Symptom Input**: Users can select their symptoms from a predefined list.
2. **Disease Prediction**: Uses **SVM** for predicting the probable disease.
3. **Medical Recommendations**: Provides recommended treatments, precautions, diet suggestions, and workout routines for the predicted disease.
4. **Model Training**: Utilizes **SVM**, **Random Forest**, and **KNN** classifiers for disease prediction.
5. **Pickle Integration**: Saves and loads the trained SVM model using **pickle**.
6. **Streamlit UI**: Interactive web-based user interface for inputting symptoms and viewing recommendations.

---

### Requirements

#### Libraries
Make sure the following Python libraries are installed:
- **Streamlit**: For building the web app UI
- **pandas**: For handling and displaying data
- **scikit-learn**: For machine learning models
- **numpy**: For array handling
- **pickle**: For saving and loading machine learning models

To install the required libraries, run:
```bash
pip install streamlit pandas scikit-learn numpy
```

---

### Folder Structure

```
|-- app.py                  # Main Python file containing the Streamlit application
|-- training_data.csv        # Dataset used for training the model
|-- medications.csv          # CSV file with medication recommendations
|-- precautions_df.csv       # CSV file with necessary precautions
|-- diets.csv                # CSV file with diet suggestions
|-- description.csv          # CSV file containing disease descriptions
|-- workout_df.csv           # CSV file containing workout recommendations
|-- svm.pkl                  # Pickle file storing the trained SVM model
|-- README.md                # Documentation file
|-- Stethescope.jpg          # Image used in the app
```

---

### How to Run the App

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/medical-recommendation-system.git
   cd medical-recommendation-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   To start the app, run the following command in the terminal:
   ```bash
   streamlit run app.py
   ```

4. **Access the app**:
   After running the command, open your browser and navigate to `http://localhost:8501`.

---

### Application Workflow

1. **User Input**:
   - Users can enter their name and select symptoms from a list of predefined symptoms.

2. **Disease Prediction**:
   - After selecting the symptoms, the system predicts the disease using the **SVM** model, which has been trained using the **training_data.csv** dataset.

3. **Medical Recommendations**:
   - Once the disease is predicted, the app provides the following:
     - Description of the disease.
     - Recommended medications.
     - Necessary precautions.
     - Suggested diet.
     - Recommended workout routines.

4. **Pickle Integration**:
   - The trained **SVM** model is saved to a file using **pickle**. When predictions are needed, the model is loaded from the **svm.pkl** file to predict diseases based on the input symptoms.

---

### Machine Learning Models

Three machine learning models are used in this system:
1. **Support Vector Machine (SVM)**:
   - Kernel: `linear`
   - Used for prediction in the app.
   
2. **Random Forest**:
   - Number of trees: 100
   - Can be used for prediction, but **SVM** is the primary model for predictions in this system.

3. **K-Nearest Neighbors (KNN)**:
   - Number of neighbors: 5
   - Another classifier model included in the system for testing purposes.

---

### How to Train the Models

The models are trained on a dataset of diseases and their associated symptoms. You can retrain the models by modifying the **app.py** script and re-running it to save a new model using **pickle**.

To train the SVM model and save it to a file:
```python
# Train the model
svm = SVC(kernel='linear')
svm.fit(x_train, y_train)

# Save the model
import pickle
with open("svm.pkl", 'wb') as f:
    pickle.dump(svm, f)
```

---

### Customization

#### Adding New Symptoms
If you want to add new symptoms to the system, you can update the `symptoms_dict` dictionary in the **app.py** script:
```python
symptoms_dict = {
    "Itching": 0,
    "Skin Rash": 1,
    "Nodal Skin Eruptions": 2,
    # Add new symptoms here
}
```

#### Modifying Medical Recommendations
To update the medical recommendations (medications, precautions, diets, and workout routines), you can edit the CSV files:
- **medications.csv**
- **precautions_df.csv**
- **diets.csv**
- **workout_df.csv**

---

### Known Limitations
1. **Limited Symptom List**: The app uses a predefined list of symptoms. Users can only select symptoms that are present in the system.
2. **Static Recommendations**: The recommendations are static and based on the content in the CSV files. They are not dynamically generated.
3. **Model Performance**: The accuracy of the predictions depends on the quality of the training data. Further improvements can be made with a larger and more diverse dataset.

---

### Future Enhancements
1. **User-Defined Symptoms**: Allow users to input free-text symptoms and map them to predefined symptoms.
2. **Multiple Model Support**: Provide an option for users to select different machine learning models for prediction.
3. **Improved User Interface**: Enhance the UI with more features and better visuals.

---

### License
This project is open-source and licensed under the MIT License.

---

Enjoy using the **Medical Recommendation System**!
