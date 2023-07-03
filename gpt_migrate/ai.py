from langchain.chat_models import ChatOpenAI
from config import OPENAI_API_KEY
import os
import openai
import time
import backoff
from utils import parse_code_string

openai.api_key = os.getenv("OPENAI_API_KEY")

if "AZURE_OPENAI_ENDPOINT" in os.environ:
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_type = 'azure'
    openai.api_version = os.getenv("AZURE_OPENAI_VERSION") or '2023-03-15-preview'
    deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-4-32k"


class AI:
    def __init__(self, model="gpt-4-32k", temperature=0.1, max_tokens=10000):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model
        try:
            _ = ChatOpenAI(model_name=model) # check to see if model is available to user
        except Exception as ex:
            print(ex)
            self.model_name = "gpt-3.5-turbo"


    @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.ServiceUnavailableError))
    def write_code(self, prompt):
        message=[{"role": "user", "content": str(prompt)}] 
        response = openai.ChatCompletion.create(
            messages=message,
            stream=False,
            engine=deployment_id,
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        if response["choices"][0]["message"]["content"].startswith("INSTRUCTIONS:"):
            return ("INSTRUCTIONS:","",response["choices"][0]["message"]["content"][14:])
        else:
            code_triples = parse_code_string(response["choices"][0]["message"]["content"])
            return code_triples


    @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.ServiceUnavailableError))
    def run(self, prompt):
        message=[{"role": "user", "content": str(prompt)}] 
        response = openai.ChatCompletion.create(
            messages=message,
            stream=True,
            engine=deployment_id,
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        chat = ""
        for chunk in response:
            delta = chunk["choices"][0]["delta"]
            msg = delta.get("content", "")
            chat += msg
        return chat
    