from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder-v2:16b-lite-instruct-q4_K_M"
    db_path: str = "/data/sales.db"
    chroma_path: str = "/data/chroma"
    app_secret_key: str = "change-me"
    log_level: str = "INFO"

    model_config = {"env_file": ".env"}


settings = Settings()
