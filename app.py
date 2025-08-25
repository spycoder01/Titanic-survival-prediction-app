import streamlit as st
import pickle

# Load saved pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title("Titanic Survival Prediction App")

st.write("Enter passenger details below and see if they would survive.")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Passenger Fare", 0, 500, 50)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Predict button
if st.button("Predict Survival"):
    input_data = [[pclass, sex, age, sibsp, parch, fare, embarked]]
    prediction = pipe.predict(input_data)[0]
    prob = pipe.predict_proba(input_data)[0][1]  # probability of survival

    if prediction == 1:
        st.success(f"✅ Survived! (Probability: {prob*100:.2f}%)")
    else:
        st.error(f"❌ Did not survive. (Probability: {prob*100:.2f}%)")
