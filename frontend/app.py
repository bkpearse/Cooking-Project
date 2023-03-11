import streamlit as st
import requests
import os

url = os.environ.get('QUERY_URL', 'http://localhost:8080/what-to-eat')

# Title of the page.
st.title('What can I cook?', anchor=None)

# Load the picture
uploaded_file = st.file_uploader("Upload a picture.", ['png', 'jpg'] )
if uploaded_file is not None:
    files = {"file": uploaded_file.getvalue()}
    # files = {'upload_file': uploaded_file}
    values = {'DB': 'photcat', 'OUT': 'csv', 'SHORT': 'short'}

    if st.button('What can I eat?', type="primary"):
        st.write('Hurry I am hungry ...')
        r = requests.post(url, files=files, data=values)

        st.write(r.json())
# Send the picture

# Receive response

# Display results.
