import os
from typing import List, Dict

from groq import Groq
import ollama

import config

GROUNDED_SYSTEM_PROMPT = (
    "You are a knowledgeable Arabic-speaking assistant. Answer the "
    "question using the provided context, in clear, natural Modern "
    "Standard Arabic — even if the question was asked in English. Always "
    "answer in Arabic, never in English, regardless of what language the "
    "question or the context is written in.\n\n"
    "Write in flowing, natural Arabic prose. Do not start sentences with "
    "phrases like 'according to source 1' or 'the context states' — "
    "explain the concept directly as if you already know it.\n\n"
    "Keep the tone confident, clear, and conversational. Avoid unnecessary "
    "hedging, and don't pad the answer with restated questions or filler "
    "openers."
)

FALLBACK_SYSTEM_PROMPT = (
    "The retrieved documents do not contain information relevant to this "
    "question. Answer using your own general knowledge instead, in clear, "
    "natural Modern Standard Arabic — even if the question was asked in "
    "English. Always answer in Arabic, never in English, regardless of "
    "what language the question was asked in.\n\n"
    "Talk about it the way you naturally would in conversation — like "
    "someone explaining a topic they're genuinely familiar with, not "
    "reciting a textbook entry. Skip any mention of documents, sources, or "
    "not finding something — just dive straight into the explanation as if "
    "the question was simply asked to you directly. Be direct, be "
    "confident, and let the answer read like a real person's response."
)

TRANSLATION_SYSTEM_PROMPT = (
    "You are a professional Arabic-to-English translator. Translate the "
    "given Arabic text into clear, natural English. Preserve the meaning "
    "faithfully — do not summarize, shorten, add commentary, or explain "
    "anything. Output ONLY the English translation, nothing else — no "
    "preamble like 'Here is the translation:', no notes, no quotation "
    "marks around the result."
)


def _build_context(chunks: List[Dict], max_chars: int = None) -> str:
    max_chars = max_chars or config.MAX_CONTEXT_CHARS
    pieces = []
    total_len = 0

    for i, chunk in enumerate(chunks, start=1):
        block = f"[Source {i}: {chunk['source']}]\nText: {chunk['text']}"
        if total_len + len(block) > max_chars:
            break
        pieces.append(block)
        total_len += len(block)

    return "\n\n".join(pieces)


# ---- Provider backends ----

def _call_groq(system_prompt: str, user_prompt: str, temperature: float) -> str:
    api_key = os.environ.get(config.GROQ_API_KEY)
    if not api_key:
        raise EnvironmentError(
            f"Set the {config.GROQ_API_KEY} environment variable with your Groq API key."
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=config.GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content

# Custom changed Ollama model setting
def _call_ollama(system_prompt: str, user_prompt: str, temperature: float) -> str:
    client = ollama.Client(host=config.OLLAMA_BASE_URL)

    _options = {
        "temperature": temperature,
        "top_k": config.OLLAMA_TOP_K,
        "top_p": config.OLLAMA_TOP_P,
        "min_p": config.OLLAMA_MIN_P,
        "repeat_penalty": config.OLLAMA_REPEAT_PENALTY,
        "repeat_last_n": config.OLLAMA_REPEAT_LAST_N,
        "num_ctx": config.OLLAMA_NUM_CTX,
        "num_predict": config.OLLAMA_NUM_PREDICT,
    }
    if config.OLLAMA_SEED is not None:
        _options["seed"] = config.OLLAMA_SEED

    try:
        response = client.chat(
            model=config.OLLAMA_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options=_options,
        )
    except Exception as e:
        raise ConnectionError(
            f"Could not reach Ollama at {config.OLLAMA_BASE_URL}. "
            f"Make sure Ollama is running and the model "
            f"'{config.OLLAMA_MODEL_NAME}' is pulled. Original error: {e}"
        )
    return response["message"]["content"]
# Default Ollama model setting
# def _call_ollama(system_prompt: str, user_prompt: str, temperature: float) -> str:
#     client = ollama.Client(host=config.OLLAMA_BASE_URL)
#     try:
#         response = client.chat(
#             model=config.OLLAMA_MODEL_NAME,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ],
#             options={"temperature": temperature},
#         )
#     except Exception as e:
#         raise ConnectionError(
#             f"Could not reach Ollama at {config.OLLAMA_BASE_URL}. "
#             f"Make sure Ollama is running and the model "
#             f"'{config.OLLAMA_MODEL_NAME}' is pulled. Original error: {e}"
#         )
#     return response["message"]["content"]


def _generate(system_prompt: str, user_prompt: str, temperature: float) -> str:
    if config.LLM_PROVIDER == "ollama":
        return _call_ollama(system_prompt, user_prompt, temperature)
    return _call_groq(system_prompt, user_prompt, temperature)


# ---- Public functions (unchanged signatures) ----

def answer_query(query: str, chunks: List[Dict], context_found: bool = True) -> str:
    if context_found:
        system_prompt = GROUNDED_SYSTEM_PROMPT
        context = _build_context(chunks)
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer in Arabic:"
    else:
        system_prompt = FALLBACK_SYSTEM_PROMPT
        user_prompt = f"Question: {query}\n\nAnswer in Arabic:"

    answer = _generate(system_prompt, user_prompt, temperature=0.2)

    if not context_found:
        answer = f"{answer.rstrip()}\n"

    return answer


def translate_to_english(arabic_text: str) -> str:
    if not arabic_text or not arabic_text.strip():
        return ""

    return _generate(TRANSLATION_SYSTEM_PROMPT, arabic_text, temperature=0.0)