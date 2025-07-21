import streamlit as st
import pandas as pd
import joblib

# --- Load ML Model and Vectorizer ---
model = joblib.load("course_predictor_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# --- Load Dataset ---
df = pd.read_csv("indian_food.csv")
df = df.fillna("Unknown")

# --- Streamlit App ---
st.title("Smart Indian Recipe Recommender (ML-powered)")

# Input box for ingredients
user_input = st.text_input("Enter ingredients you have (comma-separated):")

if user_input:
    # Clean input
    user_input_clean = user_input.lower()

    # Transform input using vectorizer
    X_input = vectorizer.transform([user_input_clean])

    # Predict course using ML model
    predicted_course = model.predict(X_input)[0]

    st.success(f"Suggested Course: {predicted_course}")

    # Filter recipes by predicted course
    filtered_recipes = df[df['course'].str.lower() == predicted_course.lower()]

    # Display results
    st.subheader("Recommended Recipes:")
    if not filtered_recipes.empty:
        for _, row in filtered_recipes.head(5).iterrows():
            st.markdown(f"### {row['name']}")
            st.write(f"Ingredients: {row['ingredients']}")
            st.write(f"Diet: {row['diet']}")
            st.write(f"Prep Time: {row['prep_time']} mins | Cook Time: {row['cook_time']} mins")
            st.write(f"Flavor Profile: {row['flavor_profile']}")
            st.write(f"Region: {row['region']}")
            st.markdown("---")
    else:
        st.warning("No matching recipes found for this course.")
