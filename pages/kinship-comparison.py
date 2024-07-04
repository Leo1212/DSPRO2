import streamlit as st
import io
import os, sys
#files in parent folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prediction as facetree
import faceExtraction as faceExtraction


# Streamlit app layout
st.title("Kinship Comparison")

# Subtitle
st.header("Are they related?")

# CSS styles for centering elements
st.markdown("""
  <style>
  div.stButton {text-align:center}
  div.stSpinner > div {
    text-align:center;
    align-items: center;
    justify-content: center;
  }
  </style>""", unsafe_allow_html=True)

# Two columns for image uploads
col1, col2 = st.columns(2)

# Function to process and save face image in memory
def process_face_image(file):
    try:
        # Convert uploaded file to BytesIO
        file_bytes = io.BytesIO(file.read())

        # Extract and save face in memory
        extracted_face = faceExtraction.extract_and_save_face(file_bytes)
        return extracted_face
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

face1_buffer, face2_buffer = None, None

with col1:
    file1 = st.file_uploader("Upload First Image", type=["jpg", "png"])
    if file1 is not None:
        st.image(file1, caption='Uploaded First Image', use_column_width=True)

with col2:
    file2 = st.file_uploader("Upload Second Image", type=["jpg", "png"])
    if file2 is not None:
        st.image(file2, caption='Uploaded Second Image', use_column_width=True)

# Centered Analyse button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("Analyse"):
        if file1 is not None and file2 is not None:
            face1_buffer = process_face_image(file1)
            face2_buffer = process_face_image(file2)

            if face1_buffer and face2_buffer:
                # Display the extracted faces
                col1, col2 = st.columns(2)
                with col1:
                    st.image(face1_buffer, caption='Extracted Face from First Image')
                with col2:
                    st.image(face2_buffer, caption='Extracted Face from Second Image')

                # Show a spinner during prediction
                with st.spinner("Predicting..."):
                    predicted_label, confidence = facetree.predict(face1_buffer, face2_buffer)

                if (confidence * 100) < 15.0: 
                    st.markdown(f"<h2 style='text-align: center;'>No kinship detected</h2>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='text-align: center;padding: 0; padding-top: 1rem;'>{facetree.get_full_relationship_name(predicted_label)}</h2><h2 style='text-align: center;padding: 0;'><span style='font-size:20px;vertical-align:middle;text-align: center;'>{(confidence * 100):.2f}%</span></h2>", unsafe_allow_html=True)
            else:
                if not face1_buffer:
                    st.error("No face found in the first image.")
                if not face2_buffer:
                    st.error("No face found in the second image.")
        else:
            st.warning("Please upload both images for analysis.")