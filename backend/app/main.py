from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Notes RAG Assistant backend is running"}
