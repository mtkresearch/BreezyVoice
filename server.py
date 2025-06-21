# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from single_inference import CustomCosyVoice, get_bopomofo_rare
from cosyvoice.utils.file_utils import load_wav
from g2pw import G2PWConverter

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

# 掛載靜態文件
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    content_to_synthesize = cosyvoice.frontend.text_normalize_new(
        tts_text, 
        split=False
    )
    content_to_synthesize_bopomo = get_bopomofo_rare(content_to_synthesize, bopomofo_converter)
    model_output = cosyvoice.inference_sft(content_to_synthesize_bopomo, spk_id)
    return StreamingResponse(generate_data(model_output))

@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))

@app.post("/add_speaker")
async def add_speaker(
    spk_id: str = Form(),
    prompt_wav: UploadFile = File(),
    prompt_text: str = Form(None)
):
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"/tmp/{prompt_wav.filename}"
        with open(temp_file_path, "wb") as buffer:
            content = await prompt_wav.read()
            buffer.write(content)
        
        # Process the saved file
        prompt_speech_16k = load_wav(temp_file_path, 16000)
        
        if not prompt_text:
            from single_inference import transcribe_audio
            prompt_text = transcribe_audio(temp_file_path)
        
        spk_info = cosyvoice.cal_spk_info(temp_file_path, prompt_text)
        cosyvoice.add_spk(spk_id, spk_info)
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        return {"status": "success", "message": f"Speaker {spk_id} added successfully"}
    except Exception as e:
        # Clean up the temporary file in case of error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return {"status": "error", "message": str(e)}

@app.post("/remove_speaker")
async def remove_speaker(spk_id: str = Form()):
    try:
        cosyvoice.remove_spk(spk_id)
        return {"status": "success", "message": f"Speaker {spk_id} removed successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/get_speakers")
async def get_speakers():
    try:
        # 這裡需要實現獲取說話者列表的邏輯
        # 暫時返回空列表
        return list(cosyvoice.list_avaliable_spks())
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='models',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    try:
        bopomofo_converter = G2PWConverter()
        cosyvoice = CustomCosyVoice(args.model_dir)
    except Exception:
        raise TypeError('no valid model_type!')
    uvicorn.run(app, host="0.0.0.0", port=args.port)