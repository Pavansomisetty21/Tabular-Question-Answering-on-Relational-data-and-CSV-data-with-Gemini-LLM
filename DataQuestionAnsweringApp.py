import streamlit as st
import pandas as pd
import google.generativeai as gemini
import re
import json
import io

# Streamlit app
st.title('Data Question Answering App with Gemini')

# Input for Gemini API Key
gemini_api_key = st.text_input("Enter your Gemini API key:", type="password")

if gemini_api_key:
    # Set up Google API Key for Gemini
    gemini.configure(api_key=gemini_api_key)

    # Choose data input method
    input_method = st.radio("Choose your data input method:", ["Upload CSV File", "Enter JSON Data"])

    if input_method == "Upload CSV File":
        # Upload CSV file
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

        if uploaded_file is not None:
            # Load the CSV file into a DataFrame
            df = pd.read_csv(uploaded_file)
            st.write("Data preview:")
            st.write(df.head())

    elif input_method == "Enter JSON Data":
        # Input for tabular data
        st.header("Enter Tabular Data")
        data_input = st.text_area("Paste your tabular data as JSON:", height=300)

        if data_input:
            try:
                # Convert JSON-like string to dictionary
                data_dict = json.loads(data_input)
                # Create DataFrame
                df = pd.DataFrame(data_dict)
                st.write("Data preview:")
                st.write(df.head())
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please enter valid JSON data.")

    if 'df' in locals():  # Check if df has been defined
        # User input for question
        question = st.text_input("Ask a question about the data:")

        if st.button('Get Answer with Gemini'):
            if question:
                # Convert the DataFrame to a string to use as context
                context = df.to_string(index=False)
                model = gemini.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(f'Answer the following question based on the data: {context}\n\nQuestion: {question}')
                answer = response.text
                cleaned_response = re.sub(r'\*', '', answer)
                st.write(f"**Answer:** {cleaned_response}")
else:
    st.warning("Please enter your Gemini API key to use Gemini features.")
