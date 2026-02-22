from fastapi import FastAPI,UploadFile,File,Form,Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from modules.load_vectorstore import load_vectorstore
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from logger import logger
import os
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from fastapi.responses import JSONResponse

app=FastAPI(title="RagBot2.0")

# allow frontend

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def catch_exception_middleware(request:Request,call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("UNHANDLED EXCEPTION")
        return JSONResponse(status_code=500,content={"error":str(exc)})
    
@app.post("/upload_pdfs/")
async def upload_pdfs(files:List[UploadFile]=File(...)):
    try:
        logger.info(f"recieved {len(files)} files")
        load_vectorstore(files)
        logger.info("documents added to chroma")
        return {"message":"Files processed and vectorstore updated"}
    except Exception as e:
        logger.exception("Error during pdf upload")
        return JSONResponse(status_code=500,content={"error":str(e)})


# @app.post("/ask/")
# async def ask_quyestion(question:str=Form(...)):
#     try:
#         logger.info("fuser query:{question}")
#         from langchain_chroma import Chroma
#         from langchain.embeddings import HuggingFaceBgeEmbeddings
#         from modules.load_vectorstore import PERSIST_DIR
#         from langchain.chains import RetrievalQA
#         from langchain.schema import BaseRetriever

#         vectorstore=Chroma(
#             persist_directory=PERSIST_DIR,
#             embedding_function=HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L12-v2")
#         )
#         chain=get_llm_chain(vectorstore)
#         result=query_chain(chain,question)
#         logger.info("query successfull")
#         return result
#     except Exception as e:
#         logger.exception("error processing question")
#         return JSONResponse(status_code=500,content={"error":str(e)})



class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask")

    @validator("question")
    def question_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("Question must not be blank or whitespace.")
        return v.strip()


class SourceItem(BaseModel):
    source: str = Field(..., description="Source document identifier")


class AskResponse(BaseModel):
    response: str = Field(..., description="The LLM-generated answer")
    sources: List[SourceItem] = Field(default_factory=list, description="Sources used to answer")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")


# ── Endpoint ─────────────────────────────────────────────────────────────────

@app.post("/ask/", response_model=AskResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def ask_question(question: str = Form(...)):
    # ── Validate Request ──────────────────────────────────────────────────────
    try:
        request_data = AskRequest(question=question)
    except ValueError as ve:
        return JSONResponse(status_code=400, content=ErrorResponse(error=str(ve)).dict())

    try:
        logger.info(f"user query: {request_data.question}")

        from pinecone import Pinecone
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_core.documents import Document
        from langchain.schema import BaseRetriever
        from typing import List
        from modules.llm import get_llm_chain
        from modules.query_handlers import query_chain
        import os

        # 1️⃣ Pinecone + Embedding
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
        embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # 2️⃣ Embed the question
        embedded_query = embed_model.embed_query(request_data.question)

        # 3️⃣ Query Pinecone
        res = index.query(vector=embedded_query, top_k=5, include_metadata=True)
        matches = res.get("matches", [])

        if not matches:
            return AskResponse(response="No relevant documents found.", sources=[])

        # 4️⃣ Convert to LangChain Documents
        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata={"source": match["metadata"].get("source", "unknown")}
            )
            for match in matches
            if "text" in match["metadata"]
        ]

        # 5️⃣ Simple Retriever
        class SimpleRetriever(BaseRetriever):
            def __init__(self, documents: List[Document]):
                super().__init__()
                self._docs = documents

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self._docs

        retriever = SimpleRetriever(docs)

        # 6️⃣ LLM + RetrievalQA
        chain = get_llm_chain(retriever)
        raw_result = query_chain(chain, request_data.question)

        # ── Validate Response ─────────────────────────────────────────────────
        validated_result = AskResponse(
            response=raw_result.get("response", ""),
            sources=[SourceItem(source=s.get("source", "unknown")) for s in raw_result.get("sources", [])]
        )

        logger.info("query successful")
        return validated_result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error=str(e)).dict()
        )



@app.get("/test")
async def test():
    return {"message":"Testing successfull..."}
