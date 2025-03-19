from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import email

app = FastAPI()


# ✅ CORS Configuration
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["loomifyinnovations.com", "backend.loomifyinnovations.com"],  # Allow all origins (change for security)
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Include Email Router
app.include_router(email.router, prefix="/api")

@app.get("/")
def home():
    return {"message": "Welcome to the Email Processing API Updates"}
j,
