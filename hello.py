import streamlit as st

st.set_page_config(
    page_title="FaceTree",
    page_icon="🌳",
)

st.write("# Welcome to FaceTree")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Welcome to FaceTree!

    In times of crisis, families can be separated, leaving a heart-wrenching void in their lives. 
    FaceTree is dedicated to reuniting these families. Our platform, tailored specifically for the Red Cross and similar organizations, uses cutting-edge facial recognition technology to compare uploaded images against our expansive database, identifying potential familial matches.

    Whether it's a natural disaster, conflict, or any other crisis, FaceTree stands as a beacon of hope. By bridging the gap with a simple photo, we strive to bring families back together, one face at a time. Join us in this noble mission - because every family deserves to be whole again.
    """
)