import streamlit as st
from PIL import Image
import ner_app  # Import the functions from the ner_app.py file
import mysql.connector
import bcrypt
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn import tree
#import session_state
#from session_state import SessionState
import base64
from streamlit_extras.switch_page_button import switch_page

# MySQL Connection
db = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="",
    database="fraud_accounts"
)

#-----------------------------------------LOGING AND VERIFICATION--------------------------------------
def check_password(provided_password, stored_password_hash):
    return bcrypt.checkpw(provided_password, stored_password_hash)

def to_bytes(s):
    if type(s) is bytes:
        return s
    elif type(s) is str or (sys.version_info[0] < 3 and type(s) is unicode):
        return codecs.encode(s, 'utf-8')
    else:
        raise TypeError("Expected bytes or string, but got %s." % type(s))

def validate_login(email, password):
    cursor = db.cursor()
    query = "SELECT password FROM users WHERE email = %s"
    cursor.execute(query, (email,))
    stored_password_hash = cursor.fetchone()

    if stored_password_hash and bcrypt.checkpw(to_bytes(password), bytes(stored_password_hash[0])):
        return True
    return False


#CREATE AN ACCOUNT FUNCTION
def create_account(email, password):
    # Hash the provided password using bcrypt
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Insert the user's email and hashed password into the database
    cursor = db.cursor()
    query = "INSERT INTO users (email, password) VALUES (%s, %s)"
    cursor.execute(query, (email, hashed_password))
    db.commit()

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



#-----------------------------------------SESSION STATE--------------------------------------
# Initialize the session state
#session_state = SessionState()
# Access the values using the get method
#current_page = session_state.get_current_page
#user_logged_in = session_state.get_user_logged_in
#-----------------------------------------LOGIN PAGE--------------------------------------
# Page function for the login page
def login_page():
    st.title("Medicine Fraud Detection App")
    set_background('C:/fraud_platform/medicine-capsules.png')
    st.write("Welcome! Please log in or sign up to continue:")
    
    # Provide unique keys for the radio buttons
    option = st.radio("Select Option", ["Login", "Sign Up"], key="login_radio")
    
    if option == "Sign Up":
        new_email = st.text_input("New Email")
        new_password = st.text_input("New Password", type="password")

        if st.button("Sign Up"):
            create_account(new_email, new_password)
            st.success("Account created successfully! Now you can log in.")

    elif option == "Login":
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Log In"):
            if validate_login(email, password.encode('utf-8')):
                st.success("Login Successful!")
                welcome_page()
            
#-----------------------------------------WELCOME PAGE--------------------------------------
# Page function for the image upload page
def welcome_page():
    st.write("Welcome to Medicine Fraud Detection App")
    set_background('C:/fraud_platform/image.png')
    st.title("Named Entity Recognition App")

    st.sidebar.title("Navigation")
    selected_option = st.sidebar.radio("Go to", ["NER", "Fraud Detection"])

    if selected_option == "NER":
        NER_page()
    elif selected_option == "Fraud Detection":
        fraud_page()

#-----------------------------------------NER PAGE--------------------------------------
def NER_page():
        st.write("Welcome to Named Entity Recognition")
        set_background('C:/fraud_platform/image.png')
        st.sidebar.title("Navigation")
        selected_option = st.sidebar.radio("Go to", ["NER", "Fraud Detection"])

        if selected_option == "NER":
            NER_page()
        elif selected_option == "Fraud Detection":
            fraud_page()

        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Extract Text"):
                text, result = ner_app.ocr_extraction(image)
                st.write("Extracted Text:", text)

                if st.button("Display Labeled Entities"):
                    doc = ner_app.perform_named_entity_recognition(text)
                    ner_app.display_doc(doc)

                if st.button("Generate PDF Summary"):
                    detail = ner_app.details_dict(doc)
                    ner_app.create_file_txt(detail)
                    ner_app.create_summary_pdf("details.txt")
                    with open("SUMMARY.pdf", "rb") as f:
                            st.write(f.read(), format="pdf")

                if st.button("Display OCR Result"):
                    ner_app.draw_contours(image, result)

#-----------------------------------------FRAUD PAGE--------------------------------------
def fraud_page():
        st.write("Welcome to Fraud Detection")
        set_background('C:/fraud_platform/image.png')
        st.sidebar.title("Navigation")
        selected_option = st.sidebar.radio("Go to", ["NER", "Fraud Detection"])

        if selected_option == "NER":
            NER_page()
        elif selected_option == "Fraud Detection":
            fraud_page()

        # Load the dataset and prepare data
        input_file = "updated_ner_results1.xlsx"
        df = pd.read_excel(input_file, header=0)
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

        # Calculate Jaccard similarity and determine fraud status
        max_jaccard_score = -1
        max_jaccard_index = -1
        example_ner_list = [detail.get("DRUGNAME"), detail.get("TYPE"), detail.get("COMPOSITION"),
                                detail.get("SIZE"), detail.get("DOSAGE")]
        example_ner_list = [item for sublist in example_ner_list if item is not None]

        flattened_list = [item for sublist in example_ner_list for item in sublist]
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

        threshold = 0.4
        if max_jaccard_score > threshold:
                        fraud_status = "This Drug is not potentially fraudulent"
        else:
                        fraud_status = "This Drug is potentially fraudulent"

        # Display results
        st.write("Max Jaccard Similarity Score:", max_jaccard_score)
        st.write("Entities from the dataset with max similarity:", entities)
        st.write("Fraud Status:", fraud_status)

#-----------------------------------------MAIN PAGE--------------------------------------
if __name__ == "__main__":
    login_page()