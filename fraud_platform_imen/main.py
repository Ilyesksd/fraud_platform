import streamlit as st
import base64
import cv2
import numpy as np
import easyocr
from camera_input_live import camera_input_live
from streamlit_card import card
import io

from PIL import Image
import plotly.figure_factory as ff
import json
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
from tqdm import tqdm
import base64

import nltk
#nltk.download('punkt')
#nltk.download('stopwords')

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import random

from spacy import displacy
from fpdf import FPDF
from sklearn import tree
import tempfile
import pytesseract
import os 
# Create a spaCy NLP pipeline
nlp = spacy.load("model-best")

import ner_app #importing the ner_app script
#-----------------------------------------SET BACKGROUND IMAGE--------------------------------------
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def cooling_highlight(val):
    color = '#ACE5EE' if val else '#F2F7FA'
    return f'background-color: {color}'
#-----------------------------------------PAGE CONFIGURATIONS--------------------------------------
st.set_page_config(
        page_title="Main page",
)

# Define CSS styles for the sidebar
sidebar_styles = """
    .sidebar-content {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sidebar-image {
        max-width: 150px;
        display: block;
        margin: 0 auto;
    }
"""

def set_background_color(hex_color, color):
    style = f"""
        <style>
        .background-text {{
            background-color: {hex_color};
            padding: 5px; /* Adjust padding as needed */
            border-radius: 5px; /* Rounded corners */
            color: {color}; /* Text color */
        }}
        </style>
    """
    return style
#-----------------------------------------WELCOME PAGE--------------------------------------

st.title("Medicine Fraud Detection App")
set_background('C:/fraud_platform_imen/medicine-capsules.png')
st.write("Welcome to Medicine Fraud Detection App")
#-----------------------------------------SIDE BAR--------------------------------------
with st.sidebar:
    st.sidebar.markdown(f"<style>{sidebar_styles}</style>", unsafe_allow_html=True)

    st.write(
        "<div style='display: flex; justify-content: center;'>"
        "<img src='https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png' style='width: 150px;'>"
        "</div>",
        unsafe_allow_html=True
    )

    st.title("PixOCR")
    choice = st.radio("Navigation", ["Extract text from images", "Display labeled text", "Download PDF Summary", "Fraud Detection"], index=0)
    st.info("This project application helps you annotate your medicine data and detect fraud.")
    st.sidebar.success("Select an option above.")


#-----------------------------------------RADIO BUTTON CHOICES--------------------------------------

# Define containers for each choice's content
if choice == "Extract text from images":

        choice = st.radio('Choose an option', ["Capture Image with Camera", "Upload Image"])
        if choice == "Capture Image with Camera":
            st.write("# See a new image every second")
            controls = st.checkbox("Show controls")
            image = camera_input_live(show_controls=controls)
            if image is not None:
                #st.write(type(image))
                st.image(image)
                if st.button("Extract Text From Image"):
                        #convert the file to an opencv image.
                        pil_image = Image.open(image)
                        numpy_array = np.array(pil_image)
                        #opencv_image = cv2.imdecode(numpy_array, 1)
                        opencv_image = numpy_array.copy()
                        #st.write(type(opencv_image))

                        # extract text and process the result
                        text, result = ner_app.ocr_extraction(opencv_image)
                        st.session_state = text
                        # display extracted text
                        st.markdown("Here's the Extracted text:")
                        st.markdown(set_background_color("#f2f7fa", 'black'), unsafe_allow_html=True)
                        styled_text = f"<div class='background-text'>{text}</div>"
                        st.markdown(styled_text, unsafe_allow_html=True)

                        # draw contours on the image
                        img = ner_app.draw_contours(opencv_image, result)

                        # display image with contours
                        st.markdown("Here's the image with contours on the text detected:")
                        st.image(img, channels="BGR")
                        
                          

        elif choice == "Upload Image":
            st.title("Upload Your Image")
            uploaded_file = st.file_uploader("Choose an image file", type=(["jpg", "png", "jpeg"]))

            if uploaded_file is not None:
                #convert the file to an opencv image.
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)

                # display the uploaded image
                st.image(opencv_image, channels="BGR")

                if st.button("Extract Text From Image"):
                    # extract text and process the result
                    text, result = ner_app.ocr_extraction(opencv_image)
                    st.session_state = text
                    # display extracted text
                    st.markdown("Here's the Extracted text:")
                    st.markdown(set_background_color("#f2f7fa", 'black'), unsafe_allow_html=True)
                    styled_text = f"<div class='background-text'>{text}</div>"
                    st.markdown(styled_text, unsafe_allow_html=True)

                    # draw contours on the image
                    img = ner_app.draw_contours(opencv_image, result)
                    # display image with contours
                    st.markdown("Here's the image with contours on the text detected:")
                    st.image(img, channels="BGR")



elif choice == "Display labeled text":
    st.title('Named Entity Recognition')
    #text = '"Tretinoin Gel USP 0.1% wlw A-Ret" Gel 0.1% wlwMENARINI20g"'
    text = st.session_state
    if st.button("Perform Named Entity Recognition"):
                doc = ner_app.perform_named_entity_recognition(text)
                st.session_state = doc
                html = ner_app.display_doc(doc)
                if html is not None : 
                    html_string = f"<h3>{html}</h3>"
                    # display annotated text
                    st.markdown("Here's the Annotated text:")
                    st.markdown(set_background_color("#f2f7fa", 'black'), unsafe_allow_html=True)
                    styled_text = f"<div class='background-text'>{html_string}</div>"
                    st.markdown(styled_text, unsafe_allow_html=True)
                else : 
                    st.write('none')

elif choice == "Download PDF Summary":
    st.title("Download PDF Summary")
    if st.button("Generate PDF Summary"):
        doc = st.session_state
        detail = ner_app.details_dict(doc)
        ner_app.create_file_txt(detail)
        ner_app.create_summary_pdf("C:/fraud_platform_imen/details.txt")            
        with open("SUMMARY.pdf", "rb") as f:
            pdf_bytes = f.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            # Embedding PDF in HTML
            pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
            # Displaying File
            st.markdown(pdf_display, unsafe_allow_html=True)
            st.session_state = detail
            st.download_button("Download PDF Summary", data=pdf_bytes, file_name="SUMMARY.pdf", mime="application/pdf")
           
            

if choice == "Fraud Detection":
    st.title("Fraud Detection with Jaccard Similarity")
    st.write('Upload the medecine data that we will compare to the text extracted : \n')
    file = st.file_uploader("Upload Your Dataset")
    if file is not None and file.name.split(".")[-1] == "csv":
        df = pd.read_csv(file)
        st.write("Uploaded file is a CSV file.")

    elif file is not None and file.name.split(".")[-1]  in ["xls", "xlsx"]:
        df = pd.read_excel(file)
        st.write("Uploaded file is an excel file.")
        # or we can do : file = st.file_uploader("Upload Your Dataset")
    else:
        st.write("Please Upload a CSV or an Excel file")
    st.dataframe(df)
    st.write('Click to perform Data Preparation : \n')
    if st.button("Data Preparation and Calculation of max Jaccard Score"):
        #Data preparation
        # Replace NaN values in the 'Dosage' column with a space
        df['DOSAGE'] = df['DOSAGE'].fillna(' ')
        # Replace NaN values in the 'Dosage' column with a space
        df['DRUGNAME'] = df['DRUGNAME'].fillna(' ')
        # Replace NaN values in the 'Dosage' column with a space
        df['SIZE'] = df['SIZE'].fillna(' ')
        # Replace NaN values in the 'Dosage' column with a space
        df['COMPOSITION'] = df['COMPOSITION'].fillna(' ')
        # Replace NaN values in the 'Dosage' column with a space
        df['TYPE'] = df['TYPE'].fillna(' ')
        st.dataframe(df)
        # Calculate Jaccard similarity and determine fraud status
        max_jaccard_score = -1
        max_jaccard_index = -1
        
        doc = st.session_state
        detail = ner_app.details_dict(doc)
        detail = ner_app.details_dict(doc)
        example_ner_list = [detail.get("DRUGNAME"), detail.get("TYPE"), detail.get("COMPOSITION"),
                                detail.get("SIZE"), detail.get("DOSAGE")]

        # Flatten the list of lists and remove None values
        flattened_list = [item for sublist in example_ner_list if sublist for item in sublist]
        example_tokens = word_tokenize(' '.join(map(str, flattened_list)))
        for index, row in df.iterrows():
                base_ner_list = [row['DRUGNAME'], row['TYPE'], row['COMPOSITION'], row['SIZE'], row['DOSAGE']]
                base_tokens = word_tokenize(' '.join(map(str, base_ner_list)))

                # Remove stopwords
                stop_words = set(stopwords.words("english"))
                filtered_base_tokens = [token for token in base_tokens if token.lower() not in stop_words]
                filtered_example_tokens = [token for token in example_tokens if token.lower() not in stop_words]
                # Calculate Jaccard similarity score
                jaccard_similarity = ner_app.ner_list_similarity_jaccard(filtered_base_tokens, filtered_example_tokens)

                if jaccard_similarity > max_jaccard_score:
                            max_jaccard_score = jaccard_similarity
                            max_jaccard_index = index

        if max_jaccard_index != -1:
                    entities = df.loc[max_jaccard_index]

        threshold = 0.8
        temp=False
        if max_jaccard_score > threshold:
                    fraud_status = "This Drug is not potentially fraudulent"
        else:
                    fraud_status = "This Drug is potentially fraudulent"
                    temp=True
        # Display result
        st.write(
                    "<span style='font-weight: bold; font-size: 20px;'>Max Jaccard Similarity Score: </span>"
                    "<br>",
                    unsafe_allow_html=True
                )

        #dsplay max jaccard score
        st.markdown(set_background_color("#f2f7fa", 'black'), unsafe_allow_html=True)
        styled_text = f"<div class='background-text'>{max_jaccard_score}</div>"
        st.markdown(styled_text, unsafe_allow_html=True)

        #display entities
        entities = pd.DataFrame(entities)
        st.dataframe(entities)

        if temp ==True :
            #dsplay conclusion
            st.markdown(set_background_color("#FF0000", 'White'), unsafe_allow_html=True)
            styled_text = f"<div class='background-text'>{fraud_status}</div>"
            st.markdown(styled_text, unsafe_allow_html=True)
        else: 
            st.markdown(set_background_color("#008000", 'White'), unsafe_allow_html=True)
            styled_text = f"<div class='background-text'>{fraud_status}</div>"
            st.markdown(styled_text, unsafe_allow_html=True)