import streamlit as st
import requests
from PIL import Image

# Set page config
st.set_page_config(page_title="KRCL RuleBot", page_icon="frontend/Konkan_Railway_logo.svg.png")

# Logo
try:
    logo = Image.open("frontend/Konkan_Railway_logo.svg.png")
    st.image(logo, width=50)
except FileNotFoundError:
    st.warning("Konkan Railway logo not found.")

# Title
st.title("KRCL RuleBot")
st.markdown("Ask me about General & Subsidiary Rules or Accident Manual.")

# User input
query = st.text_input("Enter your question:")

# 🔥 Correct endpoint!
BACKEND_URL = "https://final-or88.onrender.com/ask"

# Button
if st.button("Ask") and query:
    with st.spinner("Querying the backend..."):
        try:
            response = requests.post(BACKEND_URL, json={"input": query})
            if response.status_code == 200:
                data = response.json()
                if "error" in data:
                    st.error(f"Backend error: {data['error']}")
                else:
                    st.success("Answer received!")
                    st.markdown(f"### 🤖 Answer:\n{data['final_answer']}")
                    if data.get("action"):
                        st.markdown(f"**🔍 Used Tool:** {data['action']}")
                    if data.get("observation"):
                        st.markdown(f"**📄 Retrieved:**\n{data['observation']}")
            else:
                st.error(f"Backend returned status {response.status_code}")
                try:
                    st.error(response.json())
                except:
                    st.error(response.text)
        except Exception as e:
            st.error(f"Request failed: {str(e)}")
