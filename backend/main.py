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

    def stream_generate(prompt: str) -> Generator[str, None, None]:
        # Пример: просто отправляем prompt в Ollama и стримим ответ
        stream = ollama.chat(
            model="hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q6_K",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            yield chunk["message"]["content"]

    @app.post("/generate-stream")
    async def generate_stream(pdf: UploadFile = File(...)):
        # 1) прочитать PDF (коротко, без разбиения на разделы)
        text = (await pdf.read()).decode(errors="ignore")
        prompt = f"Составь алгоритм по клиническим рекомендациям:\n{text[:4000]}"

        # 2) вернуть SSE-стрим
        return StreamingResponse(
            stream_generate(prompt),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
