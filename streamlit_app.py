import streamlit as st

st.title("🎈 Kabao's Test App")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

st.write(
    "This app uses these models for classification: Naive Bayes, SVM, Random Forest, Decision Tree"
)
options = ["1", "2", "3"]
choose = st.selectbox("Pick an option:", options)

#st.button("Classify")
if st.button("Classify"): 
    st.write("You have clicked the button.")

if st.sidebar.button("New Button"):
    st.sidebar.write("oh wow!")

options2 = ["4", "5", "6", "whatever", ":partying:"]
st.sidebar.selectbox("Choose one of these:", options2)

cs = ["Naive Bayes", "SVM", "Random Forest", "Decision Tree"]
choose2 = st.selectbox("Which model are you using?", cs)




