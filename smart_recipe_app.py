import streamlit as st
import pandas as pd
import joblib

# Load ML model and vectorizer
model = joblib.load("course_predictor_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load dataset
df = pd.read_csv("indian_food.csv")
df = df.fillna("Unknown")

# App Title
st.title("Smart Indian Recipe Recommender (ML + Similarity)")

# Input
user_input = st.text_input("Enter ingredients you have (comma-separated):")

if user_input:
    # Clean user input
    user_input_clean = user_input.lower()
    user_ingredients = set(user_input_clean.replace(',', '').split())

    # Predict course using ML model
    X_input = vectorizer.transform([user_input_clean])
    predicted_course = model.predict(X_input)[0]
    st.success(f"Suggested Course: {predicted_course}")

    # Filter by predicted course
    filtered_df = df[df['course'].str.lower() == predicted_course.lower()].copy()

    # Clean ingredients from dataset
    def clean_ingredients(ing_str):
        return set(ing_str.lower().replace(',', '').split())

    filtered_df['ingredients_set'] = filtered_df['ingredients'].apply(clean_ingredients)

    # Jaccard similarity
    def jaccard(set1, set2):
        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / len(union) if union else 0

    # Calculate similarity with input
    filtered_df['similarity'] = filtered_df['ingredients_set'].apply(lambda x: jaccard(user_ingredients, x))

    # Sort by similarity
    top_matches = filtered_df.sort_values(by='similarity', ascending=False).head(5)

    # Display recommendations
    st.subheader("Recommended Recipes:")
    if not top_matches.empty:
        for _, row in top_matches.iterrows():
            st.markdown(f"### {row['name']}")
            st.write(f"Ingredients: {row['ingredients']}")
            st.write(f"Diet: {row['diet']}")
            st.write(f"Prep Time: {row['prep_time']} mins | Cook Time: {row['cook_time']} mins")
            st.write(f"Flavor Profile: {row['flavor_profile']}")
            st.write(f"Region: {row['region']}")
            st.markdown("---")
    else:
        st.warning("No matching recipes found.")
