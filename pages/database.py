import streamlit as st
import os
from PIL import Image

# Initialize session state
if 'delete_image' not in st.session_state:
    st.session_state['delete_image'] = None

# Function to save uploaded file
def save_uploaded_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, file.name), "wb") as f:
        f.write(file.getbuffer())
    return st.success("Saved file: {}".format(file.name))

# Function to delete image
def delete_image(image_name):
    if os.path.exists(os.path.join('DB', image_name)):
        os.remove(os.path.join('DB', image_name))
        st.session_state['delete_image'] = None
        st.success(f'Deleted {image_name}')

# Main app
def main():
    st.title("FaceTree DB Image Upload")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        save_uploaded_file('DB', uploaded_file)

    # Delete image if requested
    if st.session_state['delete_image']:
        delete_image(st.session_state['delete_image'])

    # Check if 'DB' directory exists
    if os.path.exists('DB'):
        # Display images
        image_files = [img_file for img_file in os.listdir('DB') if os.path.isfile(os.path.join('DB', img_file))]

        # Group images into rows of 4
        for i in range(0, len(image_files), 4):
            cols = st.columns(4)
            for col, img_file in zip(cols, image_files[i:i+4]):
                file_path = os.path.join('DB', img_file)
                col.image(Image.open(file_path), width=100, caption=img_file)
                
                # Delete button for each image
                if col.button(f'Delete {img_file}'):
                    st.session_state['delete_image'] = img_file
                    st.experimental_rerun()
    else:
        st.write("No images uploaded yet.")

if __name__ == "__main__":
    main()