import streamlit as st
#from pages.admin_utils import *
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

#from pages.admin_utils import *




import streamlit as st
from dotenv import load_dotenv
#from pages.admin_utils import *


from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
import pinecone
from langchain.vectorstores import Pinecone
import pandas as pd
from sklearn.model_selection import train_test_split


#Read PDF data
def read_pdf_data(pdf_file):
    pdf_page = PdfReader(pdf_file)
    text = ""
    for page in pdf_page.pages:
        text += page.extract_text()
    return text

#Split data into chunks
def split_data(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_text(text)
    docs_chunks =text_splitter.create_documents(docs)
    return docs_chunks

#Create embeddings instance
def create_embeddings_load_data():
    #embeddings = OpenAIEmbeddings()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

#Function to push data to Pinecone
def push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs):

    pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    return index

#*********Functions for dealing with Model related tasks...************

#Read dataset for model creation
def read_data(data):
    df = pd.read_csv(data,delimiter=',', header=None)  
    return df

#Create embeddings instance
def get_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

#Generating embeddings for our input dataset
def create_embeddings(df,embeddings):
    df[2] = df[0].apply(lambda x: embeddings.embed_query(x))
    return df

#Splitting the data into train & test
def split_train_test__data(df_sample):
    # Split into training and testing sets
    sentences_train, sentences_test, labels_train, labels_test = train_test_split(
    list(df_sample[2]), list(df_sample[1]), test_size=0.25, random_state=0)
    print(len(sentences_train))
    return sentences_train, sentences_test, labels_train, labels_test

#Get the accuracy score on test data
def get_score(svm_classifier,sentences_test,labels_test):
    score = svm_classifier.score(sentences_test, labels_test)
    return score






def read_pdf_data(pdf_file):
    pdf_page = PdfReader(pdf_file)
    text = ""
    for page in pdf_page.pages:
        text += page.extract_text()
    return text

def main():
    load_dotenv()
    st.set_page_config(page_title="Dump PDF to Pinecone - Vector Store")
    st.title("Please upload your files...📁 ")

    # Upload the pdf file
    pdf = st.file_uploader("Only PDF files allowed", type=["pdf"])

    # Extract the whole text from the uploaded pdf file
    if pdf is not None:
        with st.spinner('Wait for it...'):
            text=read_pdf_data(pdf)
            st.write("👉Reading PDF done")

            # Create chunks
            docs_chunks=split_data(text)
            #st.write(docs_chunks)
            st.write("👉Splitting data into chunks done")

            # Create the embeddings
            embeddings=create_embeddings_load_data()
            st.write("👉Creating embeddings instance done")

            # Build the vector store (Push the PDF data embeddings)
            push_to_pinecone("e5654078-a3b7-4ed1-adb8-6888c2bb4f0d","gcp-starter","project10",embeddings,docs_chunks)

        st.success("Successfully pushed the embeddings to Pinecone")


if __name__ == '__main__':
    main()
























if 'cleaned_data' not in st.session_state:
    st.session_state['cleaned_data'] =''
if 'sentences_train' not in st.session_state:
    st.session_state['sentences_train'] =''
if 'sentences_test' not in st.session_state:
    st.session_state['sentences_test'] =''
if 'labels_train' not in st.session_state:
    st.session_state['labels_train'] =''
if 'labels_test' not in st.session_state:
    st.session_state['labels_test'] =''
if 'svm_classifier' not in st.session_state:
    st.session_state['svm_classifier'] =''

 
st.title("Let's build our Model...")
 
# Create tabs
tab_titles = ['Data Preprocessing', 'Model Training', 'Model Evaluation',"Save Model"]
tabs = st.tabs(tab_titles)

# Adding content to each tab

#Data Preprocessing TAB
with tabs[0]:
    st.header('Data Preprocessing')
    st.write('Here we preprocess the data...')

    # Capture the CSV file
    data = st.file_uploader("Upload CSV file",type="csv")

    button = st.button("Load data",key="data")

    if button:
        with st.spinner('Wait for it...'):
            our_data=read_data(data)
            embeddings=get_embeddings()
            st.session_state['cleaned_data'] = create_embeddings(our_data,embeddings)
        st.success('Done!')


#Model Training TAB
with tabs[1]:
    st.header('Model Training')
    st.write('Here we train the model...')
    button = st.button("Train model",key="model")
    
    if button:
            with st.spinner('Wait for it...'):
                st.session_state['sentences_train'], st.session_state['sentences_test'], st.session_state['labels_train'], st.session_state['labels_test']=split_train_test__data(st.session_state['cleaned_data'])
                
                # Initialize a support vector machine, with class_weight='balanced' because 
                # our training set has roughly an equal amount of positive and negative 
                # sentiment sentences
                st.session_state['svm_classifier']  = make_pipeline(StandardScaler(), SVC(class_weight='balanced')) 

                # fit the support vector machine
                st.session_state['svm_classifier'].fit(st.session_state['sentences_train'], st.session_state['labels_train'])
            st.success('Done!')

#Model Evaluation TAB
with tabs[2]:
    st.header('Model Evaluation')
    st.write('Here we evaluate the model...')
    button = st.button("Evaluate model",key="Evaluation")

    if button:
        with st.spinner('Wait for it...'):
            accuracy_score=get_score(st.session_state['svm_classifier'],st.session_state['sentences_test'],st.session_state['labels_test'])
            st.success(f"Validation accuracy is {100*accuracy_score}%!")


            st.write("A sample run:")


            #text="lack of communication regarding policy updates salary, can we please look into it?"
            text="Experiencing network outages"
            st.write("***Our issue*** : "+text)

            #Converting out TEXT to NUMERICAL representaion
            embeddings= get_embeddings()
            query_result = embeddings.embed_query(text)

            #Sample prediction using our trained model
            result= st.session_state['svm_classifier'].predict([query_result])
            st.write("***Department it belongs to*** : "+result[0])
            

        st.success('Done!')

#Save model TAB
with tabs[3]:
    st.header('Save model')
    st.write('Here we save the model...')

    button = st.button("Save model",key="save")
    if button:

        with st.spinner('Wait for it...'):
             joblib.dump(st.session_state['svm_classifier'], 'modelsvm.pk1')
        st.success('Done!')