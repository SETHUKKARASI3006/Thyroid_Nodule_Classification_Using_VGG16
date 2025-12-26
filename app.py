import streamlit as st
import os

st.title('THYROID NODULE CLASSIFICATION SYSTEM')
up_img = st.file_uploader('Upload Ultrasound Image', type=['jpg', 'png', 'jpeg'])

if up_img is not None:
    with open(os.path.join(os.getcwd(), up_img.name), 'wb') as f:
        f.write(up_img.getvalue())

if st.button('Analyze'):
    os.system('streamlit run main.py')