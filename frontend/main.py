import streamlit as st
import requests
from PIL import Image

# Set page config with logo
st.set_page_config(page_title="KRCL RuleBot", page_icon="frontend/Konkan_Railway_logo.svg.png")

# Display Konkan Railway logo before title
try:
    logo = Image.open("frontend/Konkan_Railway_logo.svg.png")
    st.image(logo, width=50)
except FileNotFoundError:
    st.warning("Konkan Railway logo not found. Using default title only.")

# Title and description
st.title("KRCL RuleBot")
st.markdown("Ask me about General & Subsidiary Rules or Accident Manual.")

# Text input
query = st.text_input("Enter your question:")

# Set your deployed FastAPI backend URL
BACKEND_URL = "https://final-1-2t45.onrender.com/ask"

# Send query to FastAPI backend
if st.button("Ask") and query:
    with st.spinner("Querying the backend..."):
        try:
            response = requests.post(BACKEND_URL, json={"input": query})
            if response.status_code == 200:
                data = response.json()

                if "error" in data:
                    st.error(f"Backend error: {data['error']}")
                elif "answer" in data:
                    st.success("Answer received!")
                    st.markdown(f"### Answer:\n{data['answer']}")
                else:
                    st.warning("Unexpected response format.")
                    st.json(data)
            else:
                st.error(f"Backend returned status code {response.status_code}")
                try:
                    st.error(response.json())
                except:
                    st.error(response.text)
        except Exception as e:
            st.error(f"Request failed: {str(e)}")
