# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import shutil
import tempfile
from pathlib import Path
import os
import sys
from typing import Generator

# allow importing bot.py from same folder
sys.path.insert(0, str(Path(__file__).parent))
from bot import MedicalAssistant

app = FastAPI(title="Medical-Ollama API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # adjust to your domain/IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

assistant = MedicalAssistant()


@app.post("/generate")
async def generate_algorithm(pdf: UploadFile = File(...)):
    """Upload PDF → stream clinical algorithm in SSE format."""
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(pdf.file, tmp)
        tmp_path = tmp.name

    async def event_stream() -> Generator[str, None, None]:
        try:
            sections = assistant.load_guidelines(tmp_path)
            if not sections:
                yield "data: ❌ PDF без нужных разделов\n\n"
                return

            yield "data: 📖 Загружены разделы, генерирую алгоритм...\n\n"

            # --- реальный стриминг через MedicalAssistant ---
            import io
            buf = io.StringIO()

            # monkey-patch _generate_streaming: пишем в буфер и стримим
            original = assistant._generate_streaming

            def patched(sections, _) -> Generator[str, None, None]:
                class FakeFile:
                    def write(self, text: str):
                        for line in text.splitlines(keepends=True):
                            yield f"data: {line}"

                    def flush(self):
                        pass

                for token in original(sections, FakeFile()):
                    yield token

            for token in patched(sections, None):
                yield token

            yield "data: [DONE]\n\n"

        finally:
            os.remove(tmp_path)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )