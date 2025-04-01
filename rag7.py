import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Set Google API Key
GOOGLE_API_KEY = "AIzaSyDEbMTwl6pvaODmLCIuaszVZJe3J_R3lBA"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# PDF Path (Manually defined)
pdf_path = "orders_data_merged.pdf"  # Change this to your actual file path

st.set_page_config(page_title="📄 AI PDF Data Extractor with Graphs", layout="wide")
st.title("📄 AI PDF Data Extractor with Graphs")
st.markdown("Ask any question from your PDF and get structured answers, with graphs when relevant!")

# Check if the PDF file exists
if os.path.exists(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    pdf_text = "".join([page.extract_text() or "" for page in pdf_reader.pages])

    # Splitting text for vector search
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(pdf_text)

    # Setting up vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_texts(texts, embeddings)

    # AI Model setup
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff")

    # User input
    user_question = st.text_input("🔎 Ask a question from the PDF:")

    if user_question:
        # Answer retrieval
        answer = qa_chain.run(user_question)
        st.write("**Answer:**", answer)

        # Check if user wants a graph
        if any(keyword in user_question.lower() for keyword in ["graph", "chart", "bar chart", "pie chart", "donut chart", "line chart"]):
            extracted_data = {}
            data_pattern = r"([\w\s]+):\s*\$?([\d.,]+)"  # Extracts "Item: $123" format

            matches = re.findall(data_pattern, answer)
            for match in matches:
                key, value = match
                extracted_data[key.strip()] = float(value.replace(",", ""))

            if extracted_data:
                fig, ax = plt.subplots(figsize=(10, 6))

                if "pie" in user_question.lower() or "donut" in user_question.lower():
                    # Sorting data
                    sorted_data = dict(sorted(extracted_data.items(), key=lambda x: x[1], reverse=True))

                    # Limiting to 10 highest values
                    top_items = dict(list(sorted_data.items())[:10])

                    # Generate pie or donut chart
                    wedges, texts, autotexts = ax.pie(
                        top_items.values(),
                        labels=top_items.keys(),
                        autopct="%1.1f%%",
                        startangle=140,
                        wedgeprops={'width': 0.4 if "donut" in user_question.lower() else 1}
                    )

                    for text in texts + autotexts:
                        text.set_fontsize(10)

                    ax.set_title("Data Distribution (Donut Chart)" if "donut" in user_question.lower() else "Data Distribution (Pie Chart)")

                elif "line" in user_question.lower():
                    # Generate a sequential line chart (not time-series)
                    ax.plot(list(extracted_data.keys()), list(extracted_data.values()), marker='o', linestyle='-', color='b')
                    ax.set_title("Ingredient Prices (Line Chart)")
                    ax.set_ylabel("Price ($)")
                    ax.set_xlabel("Ingredients")
                    ax.set_xticklabels(extracted_data.keys(), rotation=45, ha="right")

                else:
                    # Generate bar chart
                    sns.barplot(x=list(extracted_data.keys()), y=list(extracted_data.values()), palette="Blues_d", ax=ax)
                    ax.set_title("Data Representation (Bar Chart)")
                    ax.set_ylabel("Values")
                    ax.set_xlabel("Categories")
                    ax.set_xticklabels(extracted_data.keys(), rotation=45, ha="right")

                st.pyplot(fig)

    else:
        st.warning("Please enter a question.")
else:
    st.error(f"🚨 The file `{pdf_path}` was not found! Please check the path.")
