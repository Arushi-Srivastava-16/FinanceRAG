import streamlit as st
import os
from utils.prepare_vectordb import get_vectorstore

def save_docs_to_vectordb(pdf_docs, upload_docs):
    """
    Save uploaded PDF documents to the 'docs' folder and create or update the vectorstore

    Parameters:
    - pdf_docs (list): List of uploaded PDF documents
    - upload_docs (list): List of names of previously uploaded documents
    """
    # Filter is the file is a new one or not. If it is, the button to process will appear
    new_files = [pdf for pdf in pdf_docs if pdf.name not in upload_docs]
    new_files_names = [pdf.name for pdf in new_files]
    if new_files and st.button("Process"):
        # Iterate only trough the new files and save them to the docs folder
        for pdf in new_files:
            pdf_path = os.path.join("docs", pdf.name)
            with open(pdf_path, "wb") as f:
                f.write(pdf.getvalue())
            st.session_state.uploaded_pdfs.extend(pdf_docs)
        # Display the processing message
        with st.spinner("Processing"):
            # Create or update the vectorstore with the newly uploaded documents
            get_vectorstore(new_files_names)
            st.success(f"{pdf.name} uploaded successfully to 'docs' folder.")