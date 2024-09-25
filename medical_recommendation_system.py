
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df=pd.read_csv("training_data.csv")


le=LabelEncoder()
le.fit(df['prognosis'])
LE= le.transform(df['prognosis'])



x=df.drop('prognosis',axis=1)
y=df['prognosis']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=20)

models={"SVM": SVC(kernel='linear'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=50),
        "KNN": KNeighborsClassifier(n_neighbors=5)}

for model_name, model in models.items():

  model.fit(x_train,y_train) #Training
  pred=model.predict(x_test) #Testing
  accuracy=accuracy_score(y_test,pred) #Accuracy
  conf_mat=confusion_matrix(y_test,pred)

#SVM
svm=SVC(kernel='linear')
svm.fit(x_train,y_train)
predic=svm.predict(x_test)
svm_accuracy=accuracy_score(y_test,predic)

#Pickle

import pickle
with open("svm.pkl",'wb') as f:
  pickle.dump(svm,f)

svm=pickle.load(open("svm.pkl",'rb'))

svm.predict(x_test.iloc[4].values.reshape(1,-1))

#x_test

symptoms_dict={"Itching":0,
"Skin Rash":1,
"Nodal Skin Eruptions":2,
"Continuous Sneezing":3,
"Shivering":4,
"Chills":5,
"Joint Pain":6,
"Stomach Pain":7,
"Acidity":8,
"Ulcers On Tongue":9,
"Muscle Wasting":10,
"Vomiting":11,
"Burning Micturition":12,
"Spotting Urination":13,
"Fatigue":14,
"Weight Gain":15,
"Anxiety":16,
"Cold Hands And Feets":17,
"Mood Swings":18,
"Weight Loss":19,
"Restlessness":20,
"Lethargy":21,
"Patches In Throat":22,
"Irregular Sugar Level":23,
"Cough":24,
"High Fever":25,
"Sunken Eyes":26,
"Breathlessness":27,
"Sweating":28,
"Dehydration":29,
"Indigestion":30,
"Headache":31,
"Yellowish Skin":32,
"Dark Urine":33,
"Nausea":34,
"Loss Of Appetite":35,
"Pain Behind The Eyes":36,
"Back Pain":37,
"Constipation":38,
"Abdominal Pain":39,
"Diarrhoea":40,
"Mild Fever":41,
"Yellow Urine":42,
"Yellowing Of Eyes":43,
"Acute Liver Failure":44,
"Fluid Overload":45,
"Swelling Of Stomach":46,
"Swelled Lymph Nodes":47,
"Malaise":48,
"Blurred And Distorted Vision":49,
"Phlegm":50,
"Throat Irritation":51,
"Redness Of Eyes":52,
"Sinus Pressure":53,
"Runny Nose":54,
"Congestion":55,
"Chest Pain":56,
"Weakness In Limbs":57,
"Fast Heart Rate":58,
"Pain During Bowel Movements":59,
"Pain In Anal Region":60,
"Bloody Stool":61,
"Irritation In Anus":62,
"Neck Pain":63,
"Dizziness":64,
"Cramps":65,
"Bruising":66,
"Obesity":67,
"Swollen Legs":68,
"Swollen Blood Vessels":69,
"Puffy Face And Eyes":70,
"Enlarged Thyroid":71,
"Brittle Nails":72,
"Swollen Extremeties":73,
"Excessive Hunger":74,
"Extra Marital Contacts":75,
"Drying And Tingling Lips":76,
"Slurred Speech":77,
"Knee Pain":78,
"Hip Joint Pain":79,
"Muscle Weakness":80,
"Stiff Neck":81,
"Swelling Joints":82,
"Movement Stiffness":83,
"Spinning Movements":84,
"Loss Of Balance":85,
"Unsteadiness":86,
"Weakness Of One Body Side":87,
"Loss Of Smell":88,
"Bladder Discomfort":89,
"Foul Smell Of Urine":90,
"Continuous Feel Of Urine":91,
"Passage Of Gases":92,
"Internal Itching":93,
"Toxic Look (Typhos)":94,
"Depression":95,
"Irritability":96,
"Muscle Pain":97,
"Altered Sensorium":98,
"Red Spots Over Body":99,
"Belly Pain":100,
"Abnormal Menstruation":101,
"Dischromic Patches":102,
"Watering From Eyes":103,
"Increased Appetite":104,
"Polyuria":105,
"Family History":106,
"Mucoid Sputum":107,
"Rusty Sputum":108,
"Lack Of Concentration":109,
"Visual Disturbances":110,
"Receiving Blood Transfusion":111,
"Receiving Unsterile Injections":112,
"Coma":113,
"Stomach Bleeding":114,
"Distention Of Abdomen":115,
"History Of Alcohol Consumption":116,
"Fluid Overload":117,
"Blood In Sputum":118,
"Prominent Veins On Calf":119,
"Palpitations":120,
"Painful Walking":121,
"Pus Filled Pimples":122,
"Blackheads":123,
"Scurring":124,
"Skin Peeling":125,
"Silver Like Dusting":126,
"Small Dents In Nails":127,
"Inflammatory Nails":128,
"Blister":129,
"Red Sore Around Nose":130,
"Yellow Crust Ooze":131,
"prognosis": 132
}

import numpy as np

# Attaching recommendations

medi=pd.read_csv("medications.csv")
prec=pd.read_csv("precautions_df.csv")
diet=pd.read_csv("diets.csv")
desc=pd.read_csv("description.csv")
workout=pd.read_csv("workout_df.csv")

#StepsToDo

def steps(final_disease):

  st.write(f'\nYou have been diagnosed with {final_disease}. Here is what you can do:\n')

  for j in range(len(desc['Disease'])):
    if desc['Disease'][j]==final_disease:
      st.write(f'✓ Description of {final_disease}:',desc['Description'][j])
      break

  for j in range(len(medi['Disease'])):
    if prec['Disease'][j]==final_disease:
      st.write(f'✓ Necessary Precautions:', prec['Precaution_1'][j], ",", prec['Precaution_2'][j], ",", prec['Precaution_3'][j], ",", prec['Precaution_4'][j]) #.strip("[]"))
    if medi['Disease'][j]==final_disease:
      st.write(f'✓ Treatment Recommendations for {final_disease}:', medi['Medication'][j].strip("[]"))
    if diet['Disease'][j]==final_disease:
      st.write(f'✓ Suggested Diet for {final_disease}:', diet['Diet'][j].strip('[]'))

  st.write(f'✓ What to do?')
  for j in range(len(workout['disease'])):
    if workout['disease'][j]==final_disease:
      st.write("  *",workout['workout'][j])

  
def predicted_disease(symptom):  
  zero_vector=np.zeros(len(symptoms_dict))
  for sym_1 in symptom:
    zero_vector[symptoms_dict[sym_1]]=1
  
  return svm.predict([zero_vector])[0]


def main(sym):
  symps=[i for i in sym[0]]
  
  symp_array=np.array(symps).reshape(-1,1)
  
  print(symp_array)
  final_disease=predicted_disease(symp_array[:,0])
  steps(final_disease)  

#STREAMLIT

st.title("Medical Recommendation System")
st.image("Stethescope.jpg")
name=st.text_input("Enter your name:")
sym=[]
if name:
    st.header(f'Welcome {name}')
    st.subheader('Please enter your symptoms')
    sym.append(st.multiselect("Select your major symptoms:",['Itching','Skin Rash','Nodal Skin Eruptions','Continuous Sneezing','Shivering','Chills','Stomach Pain','Acidity','Ulcers On Tongue','Muscle Wasting','Vomiting','Burning Micturition','Spotting Urination','Fatigue','Weight Loss','Restlessness','Lethargy','Patches In Throat','Irregular Sugar Level','Cough','High Fever','Sunken Eyes','Breathlessness','Dehydration','Indigestion','Headache','Loss Of Appetite','Back Pain','Abdominal Pain','Diarrhoea','Blurred And Distorted Vision','Chest Pain','Weakness In Limbs','Neck Pain','Dizziness','Obesity','Excessive Hunger','Extra Marital Contacts','Stiff Neck','Loss Of Balance','Passage Of Gases','Internal Itching','Depression','Irritability','Dischromic Patches','Watering From Eyes','Increased Appetite','Polyuria','Family History','Mucoid Sputum','Visual Disturbances']))
    
    if st.button("Diagnose",type='primary'):
      main(sym)
      