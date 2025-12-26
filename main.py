from image_processor import process_image
import streamlit as st
import os
from notification import send_alert
from PIL import Image

st.title('THYROID NODULE CLASSIFICATION')
st.markdown(
"""
Thyroid nodules are common, but accurate classification is vital. 
Our VGG16-based system analyzes ultrasound images to distinguish between Benign and Malignant nodules, 
providing a confidence score and alerting clinicians for high-risk cases.
"""
)

up_img = st.file_uploader('Upload Ultrasound Image', type=['jpg', 'png', 'jpeg'])

if up_img is not None:
    # Save uploaded file temporarily
    file_path = os.path.join(os.getcwd(), up_img.name)
    with open(file_path, 'wb') as f:
        f.write(up_img.getvalue())
    
    # Display Image
    image = Image.open(up_img)
    st.image(image, caption='Uploaded Ultrasound', use_column_width=True)

    if st.button('Classify'):
        label, prob = process_image(file_path)
        
        if label == "Malignant":
            st.error(f'DIAGNOSIS: MALIGNANT (High Risk)')
            st.write(f'Confidence Score: {prob}%')
            st.write('Recommendation: Immediate Fine-Needle Aspiration (FNA) biopsy recommended.')
            
            try:
                msg = f'URGENT: Malignant thyroid nodule detected. Confidence: {prob}%. Patient requires immediate biopsy referral.'
                send_alert(msg)
            except:
                st.warning("Notification service unavailable.")
                
        elif label == "Benign":
            st.success(f'DIAGNOSIS: BENIGN (Low Risk)')
            st.write(f'Confidence Score: {prob}%')
            st.write('Recommendation: Routine monitoring recommended.')
            
            try:
                msg = f'REPORT: Benign thyroid nodule detected. Confidence: {prob}%. Routine follow-up suggested.'
                # send_alert(msg) # Optional for benign
            except:
                pass
        else:
            st.write('Error in classification.')

else:
    st.write('Please upload an ultrasound image file.')
