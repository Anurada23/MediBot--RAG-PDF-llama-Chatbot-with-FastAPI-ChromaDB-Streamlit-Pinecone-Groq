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


# ── Snowflake Connection Helper ─────────────────────────────────────────────

def get_snowflake_connection():
    """Create and return a Snowflake connection."""
    try:
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse="MEDI_ANALYTICS_WH",
            database="MEDI_ANALYTICS",
            schema="PUBLIC"
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to Snowflake: {str(e)}")
        return None


def log_to_snowflake(user_id: str, model_used: str, tokens_used: int, latency_ms: float, question: str = None):
    """Log chat interaction to Snowflake."""
    conn = None
    try:
        conn = get_snowflake_connection()
        if conn:
            cs = conn.cursor()
            cs.execute("""
                INSERT INTO chat_logs (user_id, model_used, tokens_used, latency_ms, timestamp, question)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP(), %s)
            """, (user_id, model_used, tokens_used, latency_ms, question))
            conn.commit()
            cs.close()
            logger.info("Successfully logged to Snowflake")
    except Exception as e:
        logger.error(f"Failed to log to Snowflake: {str(e)}")
    finally:
        if conn:
            conn.close()


def update_doc_stats(doc_id: str, response_time: float):
    """Update document statistics in Snowflake."""
    conn = None
    try:
        conn = get_snowflake_connection()
        if conn:
            cs = conn.cursor()
            # Check if doc exists
            cs.execute("SELECT queries_count, avg_response_time FROM doc_stats WHERE doc_id = %s", (doc_id,))
            result = cs.fetchone()
            
            if result:
                # Update existing record
                queries_count, avg_response_time = result
                new_count = queries_count + 1
                new_avg = ((avg_response_time * queries_count) + response_time) / new_count
                
                cs.execute("""
                    UPDATE doc_stats 
                    SET queries_count = %s, avg_response_time = %s 
                    WHERE doc_id = %s
                """, (new_count, new_avg, doc_id))
            else:
                # Insert new record
                cs.execute("""
                    INSERT INTO doc_stats (doc_id, queries_count, avg_response_time)
                    VALUES (%s, %s, %s)
                """, (doc_id, 1, response_time))
            
            conn.commit()
            cs.close()
    except Exception as e:
        logger.error(f"Failed to update doc stats: {str(e)}")
    finally:
        if conn:
            conn.close()


# ── Middleware 

@app.middleware("http")
async def catch_exception_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("UNHANDLED EXCEPTION")
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ── Endpoints 

@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        logger.info(f"Received {len(files)} files")
        load_vectorstore(files)
        logger.info("Documents added to vectorstore")
        return {"message": "Files processed and vectorstore updated"}
    except Exception as e:
        logger.exception("Error during pdf upload")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ask/", response_model=AskResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def ask_question(question: str = Form(...), user_id: str = Form(default="anonymous")):
    start_time = time.time()
    
    # ── Validate Request 
    try:
        request_data = AskRequest(question=question)
    except ValueError as ve:
        return JSONResponse(status_code=400, content=ErrorResponse(error=str(ve)).dict())

    try:
        logger.info(f"User query: {request_data.question}")

        from pinecone import Pinecone
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_core.documents import Document
        from langchain.schema import BaseRetriever
        from typing import List

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

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # ── Validate Response 
        validated_result = AskResponse(
            response=raw_result.get("response", ""),
            sources=[SourceItem(source=s.get("source", "unknown")) for s in raw_result.get("sources", [])]
        )

        # ── Log to Snowflake
        # Estimate tokens (rough approximation: ~4 chars per token)
        tokens_used = len(request_data.question + validated_result.response) // 4
        model_used = os.getenv("LLM_MODEL_NAME", "gemini-pro")  # Adjust based on your LLM
        
        # Log chat interaction
        log_to_snowflake(
            user_id=user_id,
            model_used=model_used,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            question=request_data.question
        )

        # Update document statistics for each source
        for source in validated_result.sources:
            update_doc_stats(source.source, latency_ms)

        logger.info("Query successful")
        return validated_result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error=str(e)).dict()
        )


@app.get("/test")
async def test():
    return {"message": "Testing successful..."}


@app.get("/analytics/chat_logs")
async def get_chat_logs(limit: int = 100):
    """Retrieve recent chat logs from Snowflake."""
    conn = None
    try:
        conn = get_snowflake_connection()
        if not conn:
            return JSONResponse(status_code=500, content={"error": "Failed to connect to Snowflake"})
        
        cs = conn.cursor()
        cs.execute(f"""
            SELECT user_id, model_used, tokens_used, latency_ms, timestamp, question
            FROM chat_logs
            ORDER BY timestamp DESC
            LIMIT {limit}
        """)
        
        rows = cs.fetchall()
        cs.close()
        
        logs = [
            {
                "user_id": row[0],
                "model_used": row[1],
                "tokens_used": row[2],
                "latency_ms": row[3],
                "timestamp": str(row[4]),
                "question": row[5]
            }
            for row in rows
        ]
        
        return {"logs": logs, "count": len(logs)}
    
    except Exception as e:
        logger.exception("Error fetching chat logs")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if conn:
            conn.close()


@app.get("/analytics/doc_stats")
async def get_doc_stats():
    """Retrieve document statistics from Snowflake."""
    conn = None
    try:
        conn = get_snowflake_connection()
        if not conn:
            return JSONResponse(status_code=500, content={"error": "Failed to connect to Snowflake"})
        
        cs = conn.cursor()
        cs.execute("""
            SELECT doc_id, queries_count, avg_response_time
            FROM doc_stats
            ORDER BY queries_count DESC
        """)
        
        rows = cs.fetchall()
        cs.close()
        
        stats = [
            {
                "doc_id": row[0],
                "queries_count": row[1],
                "avg_response_time": row[2]
            }
            for row in rows
        ]
        
        return {"stats": stats, "count": len(stats)}
    
    except Exception as e:
        logger.exception("Error fetching doc stats")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if conn:
            conn.close()



@app.get("/test")
async def test():
    return {"message":"Testing successfull..."}
