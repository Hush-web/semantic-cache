import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "groq")

settings = Settings()