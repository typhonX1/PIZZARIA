import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

# Set Google API Key
GOOGLE_API_KEY = "AIzaSyAA9C_89mLP9YrWCJaCFUDsITS3ofmhQJU"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# PDF Path (Manually defined)
pdf_path = "address_data-merged.pdf"  # Ensure this file exists

st.set_page_config(page_title="PIZZARIA", layout="wide")
st.title("PIZZARIA🍕")

st.markdown("1️⃣ **Select a data visualization format below.** \n"
            "2️⃣ **Ask your question about the PDF.** \n"
            "3️⃣ **Get results in the selected format!** 🎉")

# Step 1: **Choose how to display the data FIRST**
format_choice = st.radio(
    "📊 How do you want the data to be displayed?",
    ("📋 Table", "📊 Bar Chart", "🥧 Pie Chart", "🍩 Donut Chart", "📈 Line Chart"),
    index=0  # Default to "Table" so it's never None
)

# Step 2: **Ask a question**
user_question = st.text_input("🔎 Ask your query:")

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

    # Step 3: **Process the query only if both inputs are provided**
    if user_question and format_choice:
        with st.spinner("🔍 Processing your request..."):
            answer = qa_chain.run(user_question)

        # Extract key-value pairs from the AI's response
        extracted_data = {}
        data_pattern = r"([\w\s]+):\s*\$?([\d.,]+)"  # Extracts "Item: $123" format
        matches = re.findall(data_pattern, answer)

        for match in matches:
            key, value = match
            try:
                extracted_data[key.strip()] = float(value.replace(",", ""))
            except ValueError:
                continue  # Skip invalid values

        # **Ensure extracted data exists**
        if extracted_data:
            df = pd.DataFrame(list(extracted_data.items()), columns=["Category", "Value"])

            # **Display the selected format**
            if format_choice == "📋 Table":
                st.write("### 📋 Table View")
                st.dataframe(df)

            elif format_choice == "📊 Bar Chart":
                st.write("### 📊 Bar Chart")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=df["Category"], y=df["Value"], palette="Blues_d", ax=ax)
                ax.set_ylabel("Value ($)")
                ax.set_xlabel("Categories")
                ax.set_xticklabels(df["Category"], rotation=45, ha="right")
                st.pyplot(fig)

            elif format_choice == "🥧 Pie Chart":
                st.write("### 🥧 Pie Chart")
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(df["Value"], labels=df["Category"], autopct="%1.1f%%", startangle=140)
                st.pyplot(fig)

            elif format_choice == "🍩 Donut Chart":
                st.write("### 🍩 Donut Chart")
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(df["Value"], labels=df["Category"], autopct="%1.1f%%", startangle=140, wedgeprops={"width": 0.4})
                st.pyplot(fig)

            elif format_choice == "📈 Line Chart":
                st.write("### 📈 Line Chart")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df["Category"], df["Value"], marker="o", linestyle="-", color="b")
                ax.set_ylabel("Value ($)")
                ax.set_xlabel("Categories")
                ax.set_xticklabels(df["Category"], rotation=45, ha="right")
                st.pyplot(fig)

        else:
            st.error("⚠️ No structured data found! Showing text response:")
            st.write(answer)  # Show text response if extraction fails

else:
    st.error(f"🚨 The file `{pdf_path}` was not found! Please check the path.")
