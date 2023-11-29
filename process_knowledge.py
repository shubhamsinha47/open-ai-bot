import os
import ssl
import nltk

from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter


load_dotenv()


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('averaged_perceptron_tagger')


loader = DirectoryLoader(os.getenv("DATA_DIR"), glob='**/*.txt')
docs = loader.load()

char_text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

doc_texts = char_text_splitter.split_documents(docs)

openai_embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
v_store = Chroma.from_documents(doc_texts, openai_embeddings, persist_directory=os.getenv("VECTOR_DIR"))
v_store.persist()

print(f'embedded data is saved in {os.getenv("VECTOR_DIR")}')
