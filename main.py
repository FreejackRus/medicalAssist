from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import shutil
import tempfile
from pathlib import Path
import os
import sys

# чтобы видеть medical_ollama.py из этой же папки
sys.path.insert(0, str(Path(__file__).parent))
from bot import MedicalAssistant  # импорт вашего класса

app = FastAPI(title="Medical-Ollama API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

assistant = MedicalAssistant()   # индекс создаётся при старте


@app.post("/generate")
async def generate_algorithm(pdf: UploadFile = File(...)):
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(pdf.file, tmp)
        tmp_path = tmp.name

    async def event_stream():
        try:
            sections = assistant.load_guidelines(tmp_path)
            if not sections:
                yield "data: ❌ PDF не содержит нужных разделов\n\n"
                return
            yield "data: 📖 Загружены разделы, генерирую алгоритм...\n\n"
            # _generate_streaming пишет в файл, но нам нужен поток в ответ.
            # Переопределим на генератор:
            import io
            buf = io.StringIO()
            # monkey-patch _generate_streaming, чтобы писал в buf
            original = assistant._generate_streaming
            def patched(sections, _):
                class FakeFile:
                    def write(self, text):
                        buf.write(text)
                        yield f"data: {text}\n\n"
                    def flush(self): pass
                for token in original(sections, FakeFile()):
                    yield token
            # Запускаем
            for token in patched(sections, None):
                yield token
            yield "data: [DONE]\n\n"
        finally:
            os.remove(tmp_path)

    return StreamingResponse(event_stream(), media_type="text/plain")