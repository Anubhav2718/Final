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
query = st.text_input("Enter your question:", "")

# UPDATED: Replace localhost with your deployed backend
BACKEND_URL = "https://final-1-2t45.onrender.com/ask"

# Send query to FastAPI backend
if st.button("Ask") and query:
    with st.spinner("Querying the backend..."):
        try:
            response = requests.post(BACKEND_URL, json={"input": query})
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if we got an error response
                if "error" in data:
                    st.error(f"Backend error: {data['error']}")
                    if "final_answer" in data:
                        st.markdown(f"### Response:\n{data['final_answer']}")
                else:
                    st.success("Answer received!")
                    st.markdown(f"### Final Answer:\n{data.get('final_answer', 'No answer provided.')}")
                    st.markdown(f"#### Tool Used: `{data.get('action', 'Unknown')}`")
                    st.markdown(f"#### Observation:\n{data.get('observation', 'No observation available')}")
            else:
                st.error(f"Backend returned status {response.status_code}")
                try:
                    error_data = response.json()
                    st.error(f"Error details: {error_data.get('detail', str(error_data))}")
                except:
                    st.error(f"Response: {response.text}")
                    
        except Exception as e:
            st.error(f"Failed to communicate with backend: {str(e)}")