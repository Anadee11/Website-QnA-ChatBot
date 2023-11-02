from dotenv import load_dotenv
import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
import xml.etree.ElementTree as ET

# load_dotenv()
# OPENAI_API_KEY = os.getenv('OPEN_AI_KEY')
#
# llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-3.5-turbo")
#
# memory = ConversationSummaryMemory(
#     llm=llm, memory_key="chat_history", return_messages=True
# )
#
# urls = [
#         "https://docs.dreamboat.ai/docs",
#         "https://docs.dreamboat.ai/reference/getting-started-with-your-api",
#         "https://docs.dreamboat.ai/changelog",
#         "https://docs.dreamboat.ai/docs/what-is-dreamboatai",
#         "https://docs.dreamboat.ai/docs/quick-start",
#         "https://docs.dreamboat.ai/docs/rest-api-proxy-way",
#         "https://docs.dreamboat.ai/docs/dreamboat-sdk-way",
#         "https://docs.dreamboat.ai/docs/logs-and-analytics",
#         "https://docs.dreamboat.ai/docs/prompt-management",
#         "https://docs.dreamboat.ai/docs/feedbacks",
#         "https://docs.dreamboat.ai/docs/️-logs",
#         "https://docs.dreamboat.ai/docs/request-tracing",
#         "https://docs.dreamboat.ai/docs/ab-testing-of-prompts",
#         "https://docs.dreamboat.ai/docs/️-testing-evaluation",
#         "https://docs.dreamboat.ai/docs/caching",
#         "https://docs.dreamboat.ai/docs/integrations",
#         "https://docs.dreamboat.ai/docs/rest-api",
#         "https://docs.dreamboat.ai/docs/dreamboat-modes",
#         "https://docs.dreamboat.ai/docs/supported-llms",
#         "https://docs.dreamboat.ai/docs/dreamboatai-headers",
#         "https://docs.dreamboat.ai/docs/features",
#         "https://docs.dreamboat.ai/changelog/welcome-to-dreamboat-ai"
#       ]
# loaders = UnstructuredURLLoader(urls=urls)
# data = loaders.load()
#
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
# docs = text_splitter.split_documents(data)
#
# embeddings = OpenAIEmbeddings()
# vectorStore_openAI = FAISS.from_documents(docs,embeddings)
#
# with open("faiss_store_openai.pkl", "wb") as f:
#     pickle.dump(vectorStore_openAI, f)
#
# with open("faiss_store_openai.pkl", "rb") as f:
#     vectorStore_openAI = pickle.load(f)
#
# chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorStore_openAI.as_retriever(), memory=memory)
#
# def main():
#     while True:
#         user_question = input("How can I help you? (Type 'exit' to quit): ")
#
#         if user_question.lower() == 'exit':
#             print("Goodbye!")
#             break
#
#         answer = chain({"question": user_question})
#         print(answer["answer"])
#
# if __name__ == "__main__":
#     main()


def get_final_redirected_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.history:
            final_url = '/'.join(response.url.split('/')[:3])
            return final_url
        else:
            return url
    except requests.RequestException as e:
        print(f"Error occurred while trying to reach the URL: {e}")
        return None

def extract_urls_from_sitemap(sitemap_url):
    try:
        response = requests.get(sitemap_url, timeout=10)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            namespaces = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            urls = [element.text for element in root.findall('.//sitemap:loc', namespaces)]
            return urls
        else:
            raise Exception(f"Failed to retrieve sitemap, status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error occurred while trying to fetch the sitemap: {e}")
        return []

def get_sitemap_from_base_url(base_url):
    final_base_url = get_final_redirected_url(base_url)
    if final_base_url:
        sitemap_url = final_base_url + '/sitemap.xml'
        return extract_urls_from_sitemap(sitemap_url)
    else:
        raise Exception("Could not determine the final redirected base URL.")


load_dotenv()
OPENAI_API_KEY = os.getenv('OPEN_AI_KEY')

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-3.5-turbo")

memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)

def main():
    base_url = input("Please enter the base URL of the website for extracting sitemap URLs: ")
    try:
        urls = get_sitemap_from_base_url(base_url)
        loaders = UnstructuredURLLoader(urls=urls)
        data = loaders.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        docs = text_splitter.split_documents(data)

        embeddings = OpenAIEmbeddings()
        vectorStore_openAI = FAISS.from_documents(docs,embeddings)

        with open("faiss_store_openai.pkl", "wb") as f:
            pickle.dump(vectorStore_openAI, f)

        with open("faiss_store_openai.pkl", "rb") as f:
            vectorStore_openAI = pickle.load(f)

        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorStore_openAI.as_retriever(), memory=memory)
        while True:
            user_question = input("How can I help you? (Type 'exit' to quit): ")

            if user_question.lower() == 'exit':
                print("Goodbye!")
                break

            answer = chain({"question": user_question})
            print(answer["answer"])

    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    main()
