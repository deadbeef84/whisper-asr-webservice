import os
from os import path
import importlib.metadata
from typing import BinaryIO, Union

import numpy as np
import ffmpeg
from fastapi import FastAPI, File, UploadFile, Query, applications
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from whisper import tokenizer

ASR_ENGINE = os.getenv("ASR_ENGINE", "openai_whisper")
if ASR_ENGINE == "faster_whisper":
    from .faster_whisper.core import transcribe
else:
    from .openai_whisper.core import transcribe

SAMPLE_RATE=16000
LANGUAGE_CODES=sorted(list(tokenizer.LANGUAGES.keys()))

app = FastAPI(
    title="whisper asr webservice",
    description="Whisper ASR Webservice is a general-purpose speech recognition webservice.",
    version="1.2.0",
    contact={
        "url": "https://github.com/ahmetoner/whisper-asr-webservice/"
    },
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={
        "name": "MIT License",
        "url": "https://github.com/ahmetoner/whisper-asr-webservice/blob/main/LICENCE"
    }
)

@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"

@app.post("/asr", tags=["Endpoints"])
def asr(
    task : Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),
    audio_file: UploadFile = File(...)
):
    return StreamingResponse(
        transcribe(load_audio(audio_file.file), task, language, initial_prompt),
        media_type="application/x-ndjson",
        headers={
                'Asr-Engine': ASR_ENGINE
            })

def load_audio(file: BinaryIO):
    """
    Open an audio file object and read as mono waveform.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    Parameters
    ----------
    file: BinaryIO
        The audio file like object
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    return np.frombuffer(file.read(), np.int16).flatten().astype(np.float32) / 32768.0
