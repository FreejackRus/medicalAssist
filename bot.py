# medical_ollama.py  —  работает только с Ollama (REST API)
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, TextIO
import fitz
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import ollama  # pip install ollama

# ------------------------- CONFIG -------------------------
OLLAMA_MODEL: str = "hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q6_K"          # сначала «ollama pull <name>»
EMBED_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SERVICES_FILE: Path = Path("docs/services.xlsx")
CHUNK_SIZE: int = 1_000
CHUNK_OVERLAP: int = 200
TOP_K: int = 50
MAX_INPUT_TOK: int = 128_000
SECTION_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)*)\s+([^\n]+)",  # 1.1, 2.3.4, … + title
    re.MULTILINE,
)
# ----------------------------------------------------------


class MedicalAssistant:
    def __init__(self) -> None:
        self.embeddings = self._lazy_embedder()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self.db = self._load_or_build_index()

    # ---------- EMBEDDER ----------
    def _lazy_embedder(self):
        return HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},  # можно "cuda", если есть
            encode_kwargs={"normalize_embeddings": True},
            cache_folder=".embed_cache",
        )

    # ---------- INDEX ----------
    def _load_or_build_index(self) -> FAISS:
        idx_path = SERVICES_FILE.with_suffix(".faiss")
        if idx_path.exists():
            print("📁 Loading cached FAISS index …")
            return FAISS.load_local(
                str(idx_path.parent),
                self.embeddings,
                index_name=idx_path.stem,
                allow_dangerous_deserialization=True,
            )

        print("🔄 Building index from", SERVICES_FILE)
        df = pd.read_excel(SERVICES_FILE)
        docs = [
            Document(
                page_content=f"Услуга {row['ID']}: {row['Название']}",
                metadata={
                    "source": "services",
                    "id": str(row["ID"]),
                    "name": str(row["Название"]),
                },
            )
            for _, row in df.iterrows()
        ]
        db = FAISS.from_documents(docs, self.embeddings)
        db.save_local(str(idx_path.parent), idx_path.stem)
        print("✅ Index saved to disk")
        return db

    # ---------- PDF ----------
    def load_guidelines(self, pdf_path: str) -> Dict[str, str]:
        """
        Reads a Russian clinical guideline PDF and returns a dict:
            {"1.1 Определение": "...", "1.2 Этиология": "...", ...}
        Keeps Cyrillic headings intact.
        """
        if not os.path.exists(pdf_path):
            print("❌ PDF not found")
            return {}

        doc = fitz.open(pdf_path)
        full_text = "\n".join(page.get_text("text") for page in doc)
        doc.close()

        # Grab diagnosis name from the first page
        title = re.search(
            r"(Сахарный диабет.*?|Гипертония.*?|Астма.*?)\n",
            full_text[:2000],
            re.I,
        )
        self.diagnosis_name = (
            title.group(1).strip() if title else "Неизвестное заболевание"
        )
        print(f"✅ Diagnosis: {self.diagnosis_name}")

        # Split by headings
        sections = {}
        splits = SECTION_PATTERN.split(full_text)
        # split returns [prefix, num1, title1, body1, num2, title2, body2, ...]
        for i in range(1, len(splits), 3):
            number, title, body = splits[i], splits[i + 1], splits[i + 2]
            key = f"{number} {title}".strip()
            sections[key] = body.strip()

        return sections

    # ---------- UTIL ----------
    @staticmethod
    def _trim_tokens(text: str, max_tok: int) -> str:
        max_chars = int(max_tok * 4.5)
        return (
            text[:max_chars].rsplit("\n", 1)[0]
            if len(text) > max_chars
            else text
        )

    def find_services(self, query: str, k: int = TOP_K) -> List[Document]:
        hits = self.db.similarity_search(query[:300], k=k, fetch_k=k * 3)
        seen, unique = set(), []
        for doc in hits:
            if doc.metadata["id"] not in seen:
                seen.add(doc.metadata["id"])
                unique.append(doc)
        return unique

    # ---------- GENERATE ----------
    def _generate_streaming(self, sections: Dict[str, str], file: TextIO) -> None:
        services = self.find_services(self.diagnosis_name)
        services_list = "\n".join(
            f"- ID {d.metadata['id']}: {d.metadata['name']}" for d in services
        )


        content = sections.get("guidelines", "")
        content = self._trim_tokens(content, MAX_INPUT_TOK)

        system_prompt = (
            "--think=false \n"
            "Ты — русскоязычный медицинский ассистент. "
            "Твой единственный источник информации — клинические рекомендации от пользователя. "
            "Включи ВСЕ данные: числа, единицы, препараты, дозы, критерии, сроки, исключения, нюансы. "
            "Для каждого пункта укажи точные числа, единицы, препараты, дозы, сроки, исключения."
            "Если в рекоменадциях явно написано «не применять» / «не делать» — включи это. "
            " Если в рекомендациях одно исследование исключает другое - включи это и распиши. Например, тест толерантности при сахарном диабете."
            "Используй маркированные списки и подразделы. "
            "Все ответы формируются исключительно на основе текста клинических рекомендаций "
            "Отвечай подробно c широкими обьяснениями, метриками  и по делу, структурируя текст по разделам: определение, диагностика, лечение, мониторинг, профилактика."
        )
        user_prompt = f"""
        На основании текста Клиических рекомендаций построй **максимально полный и детализированный** алгоритм по заболеванию «{self.diagnosis_name}».

        Требования:
        - Включи все числовые значения, единицы измерения, названия препаратов, дозы, критерии диагностики, сроки наблюдения.
        - Укажи **все исключения и особые случаи**, упомянутые в документе.
        - Включи ВСЕ числовые значения, единицы, названия, дозы, критерии, исключения и прямо процитируй запреты/особые случаи
        - Структурируй по разделам:

        ### I. Предварительная оценка
        - Определение заболевания
        - Эпидемиология, факторы риска
        - Клинические симптомы и признаки

        ### II. Диагностика
        1. Первичное обследование (жалобы, анамнез, осмотр)
        2. Лабораторные исследования (все показатели, нормы, исключения)
        3. Инструментальные и дополнительные методы
        4. Дифференциальная диагностика (полный список)
        5. Какие исследования исключают другие и в каких случаях (полный список и подробно когда используются)

        ### III. Лечение
        1. Немедикаментозные мероприятия
        2. Медикаментозная терапия (препараты, дозы, режим, коррекции)
        3. Процедуры / операции / вмешательства
        4. Мониторинг эффективности (частота, целевые значения)

        ### IV. Реабилитация и долгосрочное наблюдение
        - Сроки, методы, критерии ответа

        ### V. Особые ситуации и профилактика
        - Осложнения, противопоказания, экстренные состояния
        - Профилактика и диспансеризация

        ---

        **Клинические рекомендации:**
        {content}
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            stream = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                stream=True,
                think=False,
                options={"temperature": 0.6, "top_p": 0.95}

            )
            for chunk in stream:
                token = chunk["message"]["content"]
                file.write(token)
                print(token, end="", flush=True)
        except Exception as e:
            print("\n❌ Ошибка при вызове Ollama:", e)

    # ---------- RUN ----------
    def run(self):
        if not self.db:
            print("❌ Index not loaded")
            return
        print("🤖 Ready. Type PDF path or 'exit'")
        while True:
            pdf = input("📄 PDF: ").strip()
            if pdf.lower() in {"exit", "quit", "q"}:
                break
            if not os.path.exists(pdf):
                print("❌ File not found")
                continue

            sections = self.load_guidelines(pdf)
            if not sections:
                continue

            outfile = Path("testSec.txt")
            with outfile.open("w", encoding="utf-8") as f:
                for key, text in sections.items():
                    f.write(f"--- {key.upper()} ---\n{text}\n\n")
            print("✅ Raw sections saved →", outfile.absolute())
            safe = re.sub(r"[^\w\s-]", "", self.diagnosis_name).strip().replace(" ", "_")[:50]
            outfile = Path(f"test.txt")
            with outfile.open("w", encoding="utf-8") as f:
                f.write(f"# Расширенный алгоритм лечения {self.diagnosis_name}\n\n")
                self._generate_streaming(sections, f)
            print("\n✅ Saved →", outfile.absolute())


# ----------------------------------------------------------
if __name__ == "__main__":
    try:
        MedicalAssistant().run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("❌ Fatal:", e)