import os

from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings


load_dotenv()

openai_embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
v_store = Chroma(persist_directory=os.getenv("vector_dir"), embedding_function=openai_embeddings)

# model = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=v_store)
model = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=v_store, return_source_documents=True)


if __name__ == "__main__":

    while True:

        question = input("Enter your question:: ")

        if question in ['bye', 'end chat']:
            print(f"Bot:: Bye until next time")
            break
        # answer = model.run(question)
        # print(f"Bot:: {answer}")

        response = model({"query": question})
        print(f"Bot:: {response['result']}")

        for source in response['source_documents']:
            print(f"Source block:: {' '.join(source.page_content.split()[-10:])} ...")
            print(f"Source file:: {source.metadata['source']}")
