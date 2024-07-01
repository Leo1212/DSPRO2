import streamlit as st
import io
import os
import sys
import glob
from PIL import Image

# Files in parent folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prediction as facetree
import faceExtraction as faceExtraction

# Streamlit app layout
st.title("Kinship Comparison")

# Subtitle
st.header("Compare with our database")

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

# Function to save the face image in DB folder
def save_face_image(directory, image, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    image.save(os.path.join(directory, filename))

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

# Function to convert PIL image to bytes stream
def pil_image_to_byte_stream(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    img_byte_arr.seek(0)
    return img_byte_arr

# Upload section with multiple file types
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
save_to_db = st.checkbox("Save extracted face to DB")
uploaded_face_buffer = None
uploaded_filename = None

if uploaded_file is not None:
    uploaded_filename = uploaded_file.name
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    uploaded_face_buffer = process_face_image(uploaded_file)
    if uploaded_face_buffer:
        # Create three columns
        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the ratio as needed for your layout

        with col2:  # This is the center column
            st.image(uploaded_face_buffer, caption='Extracted Face', use_column_width=True)

        # Save the extracted face to DB if checkbox is checked
        if save_to_db:
            save_face_image('DB', Image.open(uploaded_face_buffer), uploaded_filename)

# Compare button
if st.button("Compare") and uploaded_face_buffer:
    db_images = glob.glob('DB/*')  # Search for all file types

    if db_images:
        with st.spinner("Comparing..."):
            results = []
            for db_image_path in db_images:
                if db_image_path.lower().endswith(('.jpg', '.jpeg', '.png')):

                    # Skip if the current DB image is the same as the uploaded image
                    if os.path.basename(db_image_path) == uploaded_filename:
                        print('Skipping ' + str(uploaded_filename))
                        continue

                    # Ensure a fresh start for each image processing
                    with open(db_image_path, 'rb') as f:
                        db_face_buffer = io.BytesIO(f.read())  # Fresh buffer
                        
                        uploaded_face_buffer.seek(0)
                        db_face_buffer.seek(0)
                        
                        predicted_label, confidence = facetree.predict(uploaded_face_buffer, db_face_buffer)
                        results.append((db_image_path, (predicted_label, confidence)))

            # Display results
            results.sort(key=lambda x: x[1], reverse=True)  # Sort results by score in descending order

            # Filter results to only include those with a score > 0.1%
            filtered_results = [result for result in results if result[1][1] > 0.1]

            # Display the filtered and sorted results
            if filtered_results:
                # Calculate how many rows we need
                num_rows = len(filtered_results) // 4 + (1 if len(filtered_results) % 4 else 0)

                for row in range(num_rows):
                    cols = st.columns(4)  # Create 4 columns
                    for i in range(4):
                        index = row * 4 + i
                        if index < len(filtered_results):
                            db_image_path, (label, score) = filtered_results[index]
                            with cols[i]:  # Use context manager to specify which column to write to
                                st.write(f"{facetree.get_full_relationship_name(label)}: {(score * 100):.2f}%")
                                image = Image.open(db_image_path)
                                st.image(image, caption=os.path.basename(db_image_path), use_column_width=True)
            else:
                st.warning("No matches found with a score above 10%.")
    else:
        st.warning("No images found in the DB folder for comparison.")
else:
    st.error("No face found in the uploaded image or no image uploaded.")