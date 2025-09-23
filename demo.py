# Copyleft 🄯 2025, Germano Castanho
# Free software under the GNU GPL v3


import time
from pathlib import Path

import gradio as gr
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders.obsidian import ObsidianLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters.markdown import MarkdownTextSplitter

_ = load_dotenv(find_dotenv())


VAULT = Path(__file__).parent / "data"
VAULT.mkdir(exist_ok=True)


MODEL = ChatOpenAI(model="chatgpt-4o-latest", temperature=0.2, top_p=0.8)


PROMPT = ChatPromptTemplate.from_template("""#### PERSONA
    
    You are Minerva's Owl, a virtual assistant specialized in scientific research. Your goal is to help researchers find relevant information in Obsidian files. You are extremely didactic, always explaining step by step the concepts and information you present.

    #### CONTEXT
    
    You were developed to save the researcher time and effort, allowing them to focus on more complex tasks. You are capable of analyzing documents, summarizing information, answering questions, providing insights, etc., always based on available data.
    
    #### KNOWLEDGE
    
    Your knowledge is broad, but limited. Therefore, you are forbidden from inventing information or making assumptions, unless requested. If questions cannot be answered through available data, inform the researcher. Stick to the data available in these documents:
    
    <documents> {documents} </ documents>
       
    #### HISTORY
    
    Consider the conversation history, so you can interact in a more contextualized manner with the researcher. Whenever possible, refer to previously mentioned excerpts, to better concatenate ideas, remaining contextualized to the interaction's theme. Here is the history:
    
    <chat_history> {chat_history} </ chat_history>
    
    #### QUESTION

    Answer the following researcher's question, using the data available in the documents, as well as the conversation history. Your response should be clear, objective and informative, always explaining step by step the concepts and information presented. Researcher's question:
    
    <query> {query} </ query>""")


# DOCUMENT LOADING
loader = ObsidianLoader(VAULT)
loaded_docs = loader.load()


# TEXT SPLITTING
splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(loaded_docs)


# VECTOR STORE
embed = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)
vectorstore = FAISS.from_documents(documents=chunks, embedding=embed)


# RETRIEVAL GENERATION
def chat_function(query, chat_history):
    docs = vectorstore.similarity_search(query["text"], k=10)
    documents = "\n\n".join([doc.page_content for doc in docs])

    response = MODEL.invoke(
        PROMPT.format(
            documents=documents,
            chat_history=chat_history,
            query=query["text"],
        )
    )

    chat_history.append(query["text"])
    chat_history.append(response.content)

    for i in range(len(response.content)):
        time.sleep(0.003)
        yield response.content[: i + 1]


demo = gr.ChatInterface(
    fn=chat_function,
    multimodal=True,
    type="messages",
    textbox=gr.MultimodalTextbox(
        sources=[],
        placeholder="Chat with the Owl...",
        stop_btn=True,
    ),
    editable=True,
    title="Minerva's Owl 🦉",
    description="Interact with Your Research",
    theme="gstaff/xkcd",
    css="""footer, .gradio-footer {
        visibility: hidden !important;
    }""",
    autoscroll=False,
)


if __name__ == "__main__":
    demo.launch()
