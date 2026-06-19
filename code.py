from youtube_transcript_api import YouTubeTranscriptApi ,TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnablePassthrough , RunnableLambda
from dotenv import load_dotenv

load_dotenv()


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
question = input("Ask a question: ")

#document load
video_id="LPZh9BOjkQs"
ytt_api = YouTubeTranscriptApi()

transcript = ytt_api.fetch(video_id)

text = " ".join(
    snippet.text for snippet in transcript
)


#text splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = splitter.create_documents([text])


#vectorstore
vector_store=FAISS.from_documents(chunks,embeddings)


#retriever
retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})
#retrieved_docs = retriever.invoke(question)

#prompt(augementation)
prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer only from the provided transcript context.
If the context is insufficient, say "I don't know".

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)
def format_docs(retrieved_docs):
    context_text="\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text
#final_prompt=prompt.invoke({"context":context_text,"question":question})
#answer=model.invoke(final_prompt)
#print(answer.content)
parallel_chain=RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})
parser=StrOutputParser()
main_chain = parallel_chain | prompt | model | parser
response = main_chain.invoke(question)
print(response)
