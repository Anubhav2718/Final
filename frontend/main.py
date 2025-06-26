import streamlit as st
import requests
from PIL import Image

# Set page config with logo
st.set_page_config(page_title="KRCL RuleBot", page_icon="Konkan_Railway_logo.svg.png")

# Display Konkan Railway logo before title
try:
    logo = Image.open("Konkan_Railway_logo.svg.png")
    st.image(logo, width=300)
except FileNotFoundError:
    st.warning("Konkan Railway logo not found. Using default title only.")

# Title and description
st.title("KRCL RuleBot")
st.markdown("Ask me about General & Subsidiary Rules or Accident Manual.")

# Text input
query = st.text_input("Enter your question:", "")

#  UPDATED: Replace localhost with your deployed backend
BACKEND_URL = "https://final-1-2t45.onrender.com/ask"

# Send query to FastAPI backend
if st.button("Ask") and query:
    with st.spinner("Querying the backend..."):
        try:
            response = requests.post(BACKEND_URL, json={"input": query})
            if response.status_code == 200:
                data = response.json()
                st.success("Answer received!")
                st.markdown(f"###  Final Answer:\n{data['final_answer']}")
                st.markdown(f"####  Tool Used: `{data['action']}`")
                st.markdown(f"####  Observation:\n{data['observation']}")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to reach backend: {e}")
