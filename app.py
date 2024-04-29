import streamlit as st
import cohere
import os

# Replace 'your-cohere-api-key' with your actual Cohere API key
api_key = os.getenv('COHERE_API_KEY')
print(f"API Key: {api_key}")

# Ensure the API key is actually retrieved; otherwise, notify the user.
if api_key is None:
    st.error("COHERE_API_KEY environment variable not found. Please set it.")
else:
    # Initialize the Cohere client with the API key
    co = cohere.Client(api_key)

def generate_rag_response_with_citations(query, documents):
    """
    Generates a response to the user query using Command-R model with RAG capability
    by referencing a set of user-uploaded documents and includes citations in the response.
    
    Parameters:
    - query (str): The user's query.
    - documents (list): A list of documents provided by the user.
    
    Returns:
    - Tuple[str, list]: The generated response and a list of citations.
    """
    # Format documents for the API
    formatted_documents = [{"title": f"doc_{i}", "snippet": doc} for i, doc in enumerate(documents)]
    
    # Call the Cohere chat endpoint with the documents for RAG
    response = co.chat(
        model="command-r",
        message=query,
        connectors=[{"id": "web-search"}]
    )   

    # Extracting text and citations from the response
    response_text = response.text
    citations = response.citations

    return response_text, citations

# Streamlit UI
st.title('RAG with Citations - Command-r')

uploaded_files = st.file_uploader("Upload documents related to your query (text files):", accept_multiple_files=True, type=['txt'])
user_query = st.text_area("Enter your query:")

if st.button('Get Answer'):
    if not user_query:
        st.write("Please enter a query to proceed.")
    elif not uploaded_files:
        st.write("Please upload at least one document to proceed.")
    else:
        # Read the content of the uploaded files
        documents = [file.getvalue().decode("utf-8") for file in uploaded_files]
        
        response, citations = generate_rag_response_with_citations(user_query, documents)
        st.write("Answer:")
        st.write(response)
        
        if citations:
            st.write("Citations:")
            for citation in citations:
                cited_text = citation['text']
                document_ids = citation['document_ids']
                # Assuming document IDs are in the format "doc_x", extract and display the cited document snippets
                for doc_id in document_ids:
                    index = int(doc_id.split('_')[-1])
                    st.write(f"- {cited_text} (from document: {documents[index]})")