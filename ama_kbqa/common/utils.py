from os import getenv
from openai import OpenAI


def get_model():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=getenv("OPENROUTER_API_KEY"),
    )


def get_embedding_model():
    pass
