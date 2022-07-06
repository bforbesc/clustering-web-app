import streamlit as st
import pandas as pd
import seaborn as sns


# Source #1: https://www.projectpro.io/recipes/add-file-uploader-widget-streamlit
from io import StringIO

# adding a file uploader

file = st.file_uploader("Please choose a file")

if file is not None:
    #To read file as bytes:
    bytes_data = file.getvalue()
    st.write(bytes_data)

    #To convert to a string based IO:
    stringio = StringIO(file.getvalue().decode("utf-8"))
    st.write(stringio)

    #To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    #Can be used wherever a "file-like" object is accepted:
    df= pd.read_csv(file)
    st.write(df)

#adding a file uploader to accept multiple CSV files

uploaded_files = st.file_uploader("Please choose a CSV file", accept_multiple_files=True)
for file in uploaded_files:
    bytes_data = file.read()
    st.write("File uploaded:", file.name)
    st.write(bytes_data)

# Source #2: https://towardsdatascience.com/how-to-write-web-apps-using-simple-python-for-data-scientists-a227a1a01582

# check box
if st.checkbox('Show dataframe'):
    st.write(df)

# select box
option = st.selectbox(
    'Which Club do you like best?',
     df['Club'].unique())

st.write('You selected: ', option)

# multiselect
options = st.multiselect(
 'What are your favorite clubs?', df['Club'].unique())
st.write('You selected:', options)