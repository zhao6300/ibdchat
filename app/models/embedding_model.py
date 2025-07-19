import json
import logging
import os
import re
import threading
from abc import ABC
from urllib.parse import urljoin

import dashscope
import numpy as np
import requests

from ollama import Client
from openai import OpenAI

# --- Utility Functions ---
# Configure logging
logging.basicConfig(
    format='%((asctime)s)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def truncate(text: str, max_length: int) -> str:
    """
    Truncate the input text to a specified maximum length.
    """
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    return text if len(text) <= max_length else text[:max_length]


def log(msg: str) -> None:
    """
    General-purpose logging function.
    """
    logging.info(msg)


def log_exception(exc: Exception, resp=None) -> None:
    """
    Log exceptions with optional response data for debugging.
    """
    if resp is not None:
        logging.error(f"Exception: {exc}, Response: {resp}")
    else:
        logging.error(f"Exception: {exc}")
    logging.exception(exc)


class Base(ABC):
    def __init__(self, key, model_name):
        pass

    def encode(self, texts: list):
        raise NotImplementedError("Please implement encode method!")

    def encode_queries(self, text: str):
        raise NotImplementedError("Please implement encode method!")

    def total_token_count(self, resp):
        try:
            return resp.usage.total_tokens
        except Exception:
            pass
        try:
            return resp["usage"]["total_tokens"]
        except Exception:
            pass
        return 0

    def encode(self, texts: list):
        raise NotImplementedError("Please implement encode method!")

    def embed_documents(self, textx: list):
        self.encode(textx)


class OpenAIEmbed(Base):
    _FACTORY_NAME = "OpenAI"

    def __init__(self, key, model_name="text-embedding-ada-002", base_url="https://api.openai.com/v1"):
        if not base_url:
            base_url = "https://api.openai.com/v1"
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name

    def encode(self, texts: list):
        # OpenAI requires batch size <=16
        batch_size = 16
        texts = [truncate(t, 8191) for t in texts]
        ress = []
        total_tokens = 0
        for i in range(0, len(texts), batch_size):
            res = self.client.embeddings.create(
                input=texts[i: i + batch_size], model=self.model_name)
            try:
                ress.extend([d.embedding for d in res.data])
                total_tokens += self.total_token_count(res)
            except Exception as _e:
                log_exception(_e, res)
        return np.array(ress), total_tokens

    def encode_queries(self, text):
        res = self.client.embeddings.create(
            input=[truncate(text, 8191)], model=self.model_name)
        return np.array(res.data[0].embedding), self.total_token_count(res)


class LocalAIEmbed(Base):
    _FACTORY_NAME = "LocalAI"

    def __init__(self, key, model_name, base_url):
        if not base_url:
            raise ValueError("Local embedding model url cannot be None")
        base_url = urljoin(base_url, "v1")
        self.client = OpenAI(api_key="empty", base_url=base_url)
        self.model_name = model_name.split("___")[0]

    def encode(self, texts: list):
        batch_size = 16
        ress = []
        for i in range(0, len(texts), batch_size):
            res = self.client.embeddings.create(
                input=texts[i: i + batch_size], model=self.model_name)
            try:
                ress.extend([d.embedding for d in res.data])
            except Exception as _e:
                log_exception(_e, res)
        # local embedding for LmStudio do not count tokens
        return np.array(ress), 1024

    def encode_queries(self, text):
        embds, cnt = self.encode([text])
        return np.array(embds[0]), cnt


class QWenEmbed(Base):
    _FACTORY_NAME = "Tongyi-Qianwen"

    def __init__(self, key, model_name="text_embedding_v2", **kwargs):
        self.key = key
        self.model_name = model_name

    def encode(self, texts: list):
        import time

        batch_size = 4
        res = []
        token_count = 0
        texts = [truncate(t, 2048) for t in texts]
        for i in range(0, len(texts), batch_size):
            retry_max = 5
            resp = dashscope.TextEmbedding.call(
                model=self.model_name, input=texts[i: i + batch_size], api_key=self.key, text_type="document")
            while (resp.get("output") is None or resp["output"].get("embeddings") is None) and retry_max > 0:
                time.sleep(10)
                resp = dashscope.TextEmbedding.call(
                    model=self.model_name, input=texts[i: i + batch_size], api_key=self.key, text_type="document")
                retry_max -= 1
            if retry_max == 0 and (resp.get("output") is None or resp["output"].get("embeddings") is None):
                msg = resp.get(
                    "message", "Retry_max reached, calling embedding model failed")
                log_exception(ValueError(f"Retry_max reached: {msg}"))
                raise RuntimeError(msg)
            try:
                embds = [[] for _ in range(len(resp["output"]["embeddings"]))]
                for e in resp["output"]["embeddings"]:
                    embds[e["text_index"]] = e["embedding"]
                res.extend(embds)
                token_count += self.total_token_count(resp)
            except Exception as _e:
                log_exception(_e, resp)
                raise
        return np.array(res), token_count

    def encode_queries(self, text):
        resp = dashscope.TextEmbedding.call(
            model=self.model_name, input=text[:2048], api_key=self.key, text_type="query")
        try:
            return np.array(resp["output"]["embeddings"][0]["embedding"]), self.total_token_count(resp)
        except Exception as _e:
            log_exception(_e, resp)


class OllamaEmbed(Base):
    _FACTORY_NAME = "Ollama"

    _special_tokens = ["<|endoftext|>"]

    def __init__(self, key, model_name, **kwargs):
        self.client = Client(host=kwargs.get("base_url")) if not key or key == "x" else Client(
            host=kwargs.get("base_url"), headers={"Authorization": f"Bear {key}"})
        self.model_name = model_name

    def encode(self, texts: list):
        arr = []
        tks_num = 0
        for txt in texts:
            for token in OllamaEmbed._special_tokens:
                txt = txt.replace(token, "")
            res = self.client.embeddings(
                prompt=txt, model=self.model_name, options={"use_mmap": True}, keep_alive=-1)
            try:
                arr.append(res["embedding"])
            except Exception as _e:
                log_exception(_e, res)
            tks_num += 128
        return np.array(arr), tks_num

    def encode_queries(self, text):
        for token in OllamaEmbed._special_tokens:
            text = text.replace(token, "")
        res = self.client.embeddings(
            prompt=text, model=self.model_name, options={"use_mmap": True}, keep_alive=-1)
        try:
            return np.array(res["embedding"]), 128
        except Exception as _e:
            log_exception(_e, res)
