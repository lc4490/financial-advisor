import streamlit as st;
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
import dotenv
import json
import yfinance as yf
import concurrent.futures
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import numpy as np
import requests
import os

st.set_page_config(
    page_title="Financial Advisor",
    # page_icon=,
    # layout="wide",  # Layout options: "centered" or "wide"
    # initial_sidebar_state="expanded",  # Sidebar options: "auto", "expanded", or "collapsed"
)

st.title("Financial Advisor")

# OPENAI
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "llama-3.1-8b-instant"

# get stock info
def get_stock_info(symbol: str):
    """
    Retrieves and formats detailed information about a stock from Yahoo Finance.

    Args:
        symbol (str): The stock ticker symbol to look up.

    Returns:
        dict: A dictionary containing detailed stock information, including ticker, name,
              business summary, city, state, country, industry, and sector.
    """
    data = yf.Ticker(symbol)
    stock_info = data.info

    properties = {
        "Ticker": stock_info.get('symbol', 'Information not available'),
        'Name': stock_info.get('longName', 'Information not available'),
        'Business Summary': stock_info.get('longBusinessSummary'),
        'City': stock_info.get('city', 'Information not available'),
        'State': stock_info.get('state', 'Information not available'),
        'Country': stock_info.get('country', 'Information not available'),
        'Industry': stock_info.get('industry', 'Information not available'),
        'Sector': stock_info.get('sector', 'Information not available')
    }

    return properties

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    """
    Generates embeddings for the given text using a specified Hugging Face model.

    Args:
        text (str): The input text to generate embeddings for.
        model_name (str): The name of the Hugging Face model to use.
                          Defaults to "sentence-transformers/all-mpnet-base-v2".

    Returns:
        np.ndarray: The generated embeddings as a NumPy array.
    """
    model = SentenceTransformer(model_name)
    return model.encode(text)

def cosine_similarity_between_sentences(sentence1, sentence2):
    """
    Calculates the cosine similarity between two sentences.

    Args:
        sentence1 (str): The first sentence for similarity comparison.
        sentence2 (str): The second sentence for similarity comparison.

    Returns:
        float: The cosine similarity score between the two sentences,
               ranging from -1 (completely opposite) to 1 (identical).

    Notes:
        Prints the similarity score to the console in a formatted string.
    """
    # Get embeddings for both sentences
    embedding1 = np.array(get_huggingface_embeddings(sentence1))
    embedding2 = np.array(get_huggingface_embeddings(sentence2))

    # Reshape embeddings for cosine_similarity function
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)

    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)
    similarity_score = similarity[0][0]
    print(f"Cosine similarity between the two sentences: {similarity_score:.4f}")
    return similarity_score

# get all companies in nyse
def get_company_tickers():
    """
    Downloads and parses the Stock ticker symbols from the GitHub-hosted SEC company tickers JSON file.

    Returns:
        dict: A dictionary containing company tickers and related information.

    Notes:
        The data is sourced from the official SEC website via a GitHub repository:
        https://raw.githubusercontent.com/team-headstart/Financial-Analysis-and-Automation-with-LLMs/main/company_tickers.json
    """
    # Check if the file already exists
    file_name = "company_tickers.json"
    if os.path.exists(file_name):
        print(f"File '{file_name}' already exists. Loading from local file...")
        with open(file_name, "r", encoding="utf-8") as file:
            company_tickers = json.load(file)
        return company_tickers
    # URL to fetch the raw JSON file from GitHub
    url = "https://raw.githubusercontent.com/team-headstart/Financial-Analysis-and-Automation-with-LLMs/main/company_tickers.json"

    # Making a GET request to the URL
    response = requests.get(url)

    # Checking if the request was successful
    if response.status_code == 200:
        # Parse the JSON content directly
        company_tickers = json.loads(response.content.decode('utf-8'))

        # Optionally save the content to a local file for future use
        with open("company_tickers.json", "w", encoding="utf-8") as file:
            json.dump(company_tickers, file, indent=4)

        print("File downloaded successfully and saved as 'company_tickers.json'")
        return company_tickers
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        return None

# process stock
def process_stock(stock_ticker: str):
    # Skip if already processed
    if stock_ticker in successful_tickers:
        return f"Already processed {stock_ticker}"

    try:
        # Get and store stock data
        stock_data = get_stock_info(stock_ticker)
        stock_description = stock_data['Business Summary']

        # Store stock description in Pinecone
        vectorstore_from_texts = PineconeVectorStore.from_documents(
            documents=[Document(page_content=stock_description, metadata=stock_data)],
            embedding=hf_embeddings,
            index_name=index_name,
            namespace=namespace
        )

        # Track success
        with open('successful_tickers.txt', 'a') as f:
            f.write(f"{stock_ticker}\n")
        successful_tickers.append(stock_ticker)

        return f"Processed {stock_ticker} successfully"

    except Exception as e:
        # Track failure
        with open('unsuccessful_tickers.txt', 'a') as f:
            f.write(f"{stock_ticker}\n")
        unsuccessful_tickers.append(stock_ticker)

        return f"ERROR processing {stock_ticker}: {e}"

# parallel processing multiple stocks at the same time
def parallel_process_stocks(tickers: list, max_workers: int = 10):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(process_stock, ticker): ticker
            for ticker in tickers
        }

        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                print(result)

                # Stop on error
                if result.startswith("ERROR"):
                    print(f"Stopping program due to error in {ticker}")
                    executor.shutdown(wait=False)
                    raise SystemExit(1)

            except Exception as exc:
                print(f'{ticker} generated an exception: {exc}')
                print("Stopping program due to exception")
                executor.shutdown(wait=False)
                raise SystemExit(1)

# RAG  
def perform_rag(query, namespace, pinecone_index, message_history):
    raw_query_embedding = get_huggingface_embeddings(query)

    top_matches = pinecone_index.query(
        vector=raw_query_embedding.tolist(),
        top_k=5,
        include_metadata=True,
        namespace=namespace
    )

    # Get the list of retrieved texts
    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    # Modify the prompt below as needed to improve the response quality
    system_prompt = f"""You are an expert at providing answers about stocks. Please answer my question provided.
    """

    # Include the existing message base in the conversation
    messages = [{"role": "system", "content": system_prompt}]
    for msg in message_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": augmented_query})

    # Call the model with streaming
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        stream=True
    )

    return stream

# CODE START

# global variables
if "first_run" not in st.session_state:
    st.session_state.first_run = True

if "namespace" not in st.session_state:
    st.session_state.namespace = None
    
if "pinecone_index" not in st.session_state:
    st.session_state.pinecone_index = None
    
if st.session_state.first_run:
    # get company tickers
    status = st.markdown("🔄 **Getting company tickers...**")
    company_tickers = get_company_tickers()

    # initialize pinecone
    status.markdown("🔄 **Initializing Pinecone vector store...**")
    pinecone_api_key = api_key=st.secrets["PINECONE_API_KEY"]
    os.environ['PINECONE_API_KEY'] = pinecone_api_key

    index_name = "stocks"
    namespace = "stock-descriptions"

    hf_embeddings = HuggingFaceEmbeddings()
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=hf_embeddings)

    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    pinecone_index = pc.Index(index_name)
    
    st.session_state.namespace = namespace
    st.session_state.pinecone_index = pinecone_index


    # Initialize tracking lists
    status.markdown("🔄 **Initializing tracking lists...**")
    successful_tickers = []
    unsuccessful_tickers = []

    # Load existing successful/unsuccessful tickers
    try:
        with open('successful_tickers.txt', 'r') as f:
            successful_tickers = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(successful_tickers)} successful tickers")
    except FileNotFoundError:
        print("No existing successful tickers file found")

    try:
        with open('unsuccessful_tickers.txt', 'r') as f:
            unsuccessful_tickers = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(unsuccessful_tickers)} unsuccessful tickers")
    except FileNotFoundError:
        print("No existing unsuccessful tickers file found")
        
    # Prepare your tickers
    status.markdown("🔄 **Preparing tickers...**")
    tickers_to_process = [company_tickers[num]['ticker'] for num in company_tickers.keys()]

    # Process them
    status.markdown("🔄 **Processing tickers...**")
    parallel_process_stocks(tickers_to_process, max_workers=10)

    status.markdown("✅ **Done! Ready to chat!**")
    st.session_state.first_run = False

# chatbot
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask the Financial Advisor"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # stream = client.chat.completions.create(
        #     model=st.session_state["openai_model"],
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # )
        response = st.write_stream(perform_rag(prompt, st.session_state.namespace, st.session_state.pinecone_index, st.session_state.messages))
    st.session_state.messages.append({"role": "assistant", "content": response})