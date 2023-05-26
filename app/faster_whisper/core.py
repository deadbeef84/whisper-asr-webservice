import json
import os
from typing import BinaryIO, Union
from io import StringIO
from threading import Lock
import torch

import whisper
from .utils import model_converter, ResultWriter, WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON
from faster_whisper import WhisperModel

model_name= os.getenv("ASR_MODEL", "base")
# model_path = os.path.join("/root/.cache/faster_whisper", model_name)
# model_converter(model_name, model_path)

if torch.cuda.is_available():
    model = WhisperModel(model_name, download_root="/root/.cache/faster_whisper", device="cuda", compute_type="float16")
else:
    model = WhisperModel(model_name, download_root="/root/.cache/faster_whisper", device="cpu", compute_type="int8")
model_lock = Lock()

def transcribe(
    audio,
    task: Union[str, None],
    language: Union[str, None],
    initial_prompt: Union[str, None]
):
    options_dict = {"task" : task}
    if language:
        options_dict["language"] = language
    if initial_prompt:
        options_dict["initial_prompt"] = initial_prompt
    with model_lock:
        segments = []
        i = 0
        segment_generator, info = model.transcribe(audio, beam_size=5, word_timestamps=True, **options_dict)
        yield json.dumps({ "language": options_dict.get("language", info.language) })
        yield "\n"
        for segment in segment_generator:
            yield json.dumps({ "start": segment.start, "end": segment.end, "words": [word._asdict() for word in segment.words] }) + "\n"
