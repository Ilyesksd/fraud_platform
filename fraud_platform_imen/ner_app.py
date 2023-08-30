import os
from PIL import Image
import streamlit as st
import cv2
import matplotlib.pyplot as plt
import json
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
from tqdm import tqdm


import nltk
nltk.download('punkt')
nltk.download('stopwords')

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import random

from spacy import displacy
from fpdf import FPDF
import numpy as np
from sklearn import tree

from PIL import Image
import pytesseract
import easyocr
reader = easyocr.Reader(['en'])
import tempfile
#--------------------------------- OCR -----------------------------------

###############OCR EXTRACTION OF TEXT AND RESULT FUNCTION##########
def ocr_extraction(IMAGE_PATH):
    reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
    result = reader.readtext(IMAGE_PATH,paragraph="False")
    text=''
    for res in result:
        text+=res[1]+''
    return text,result    #return all text and the result of each #text detected from image

############# OCR CONTOURS DETECTION FUNCTION ###########
def draw_contours(image, result):
    for detection in result:
        top_left = tuple(detection[0][0])
        bottom_right = tuple(detection[0][2])
        img = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)
    return img

#--------------------------------- Named Entity Recognition -----------------------------------
# Create a spaCy NLP pipeline
nlp = spacy.load("model-best")
# Named Entity Recognition and display functions
#@st.experimental_memo 
def perform_named_entity_recognition(text):
    doc = nlp(text)
    return doc

######COLOR GENERATOR FUNCTION ######
#@st.experimental_memo 
def color_gen(): #this function generates and returns a random color.
    random_number = random.randint(0,16777215) #16777215 ~= 256x256x256(R,G,B)
    hex_number = format(random_number, 'x')
    hex_number = '#' + hex_number
    return hex_number #generate color randomly

#####DISPLAY DOCUMENT FUNCTION########
#@st.experimental_memo 
def display_doc(doc):
    colors = {ent.label_: color_gen() for ent in doc.ents}
    options = {"ents": [ent.label_ for ent in doc.ents], "colors": colors}
    html = displacy.render(doc, style='ent', options=options, page=True, minify=True)
    return html
    #st.write(html, unsafe_allow_html=True)#display of entities recognition in text

#--------------------------------- Summary document -----------------------------------
#DETAILS EXTRACTION FUNCTION OF DOCUMENT(LABEL->ENTITIES) #
@st.experimental_memo 
def details_dict(_doc):
    Details = {}
    for ent in _doc.ents:
    #     print(ent.ents,ent.label_)
        if(ent.label_ not in Details):
            Details[ent.label_]=[str(ent.ents[0])]
        else:
            if(str(ent.ents[0]).strip() not  in Details[ent.label_] ):
                Details[ent.label_].append(str(ent.ents[0]))
    return Details #return detail label+all his entities
@st.experimental_memo 
def create_file_txt(dict_variable):
    #####
    #we must have details.txt file on our local machine
    text_file = open("C:/fraud_platform_imen/details.txt", "w")
    #####
    Details=dict_variable

    for dic in Details:
        txt=dic.upper() +' : '
        for i in range(len(Details[dic])):
            if(i<len(Details[dic])-1):
                txt+=Details[dic][i]+' , '
            else:
                txt+=Details[dic][i]
        txt+='\n'
        text_file.write(txt)#close file
    text_file.close()

#PDF SUMMARY OF THE MEDICAL REPORT FUNCTION #
@st.experimental_memo 
def create_summary_pdf(file_txt_path):
    # save FPDF() class into a
    # variable pdf
    pdf = FPDF()
    # Add a page
    pdf.add_page()# set style and size of font
    pdf.set_font("Arial", size = 15)# create a cell
    pdf.cell(200, 10, txt = "HEALTHCARE", ln = 1, align = 'C')# add another cell
    pdf.cell(200, 10, txt = "Drug Information",ln=1 , align = 'C')
    f = open(file_txt_path, "r")
    pdf.set_font("Arial", size = 10)
    for x in f:
       pdf.multi_cell(0, 5, txt = '\n'+x)
    f.close()
    # save the pdf with name .pdf
    pdf.output("C:/fraud_platform_imen/SUMMARY.pdf")
#detail = details_dict(doc)
#create_file_txt(detail)
#create_summary_pdf("details.txt")

#--------------------------------- Fraud detection -----------------------------------
@st.experimental_memo 
def ner_list_similarity_jaccard(ner_list1, ner_list2):
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_ner_list1 = [token for token in ner_list1 if token.lower() not in stop_words]
    filtered_ner_list2 = [token for token in ner_list2 if token.lower() not in stop_words]

    # Calculate Jaccard similarity
    intersection_size = len(set(filtered_ner_list1).intersection(filtered_ner_list2))
    union_size = len(set(filtered_ner_list1).union(filtered_ner_list2))

    jaccard_similarity = intersection_size / union_size

    return jaccard_similarity