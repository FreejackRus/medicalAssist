// src/App.tsx
import { useState, ChangeEvent } from "react";
import "./App.css";

function App() {
    const [text, setText] = useState<string>("");
    const [loading, setLoading] = useState<boolean>(false);

    const handleUpload = async (e: ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setText("");
        setLoading(true);

        const form = new FormData();
        form.append("pdf", file);

        const res = await fetch("/generate", {
            method: "POST",
            body: form,
        });

        const reader = res.body?.getReader();
        const decoder = new TextDecoder("utf-8");
        if (!reader) return;

        // потоковое чтение с мгновенным выводом
        const pump = async () => {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true });
                setText((prev) => prev + chunk);
            }
            setLoading(false);
        };
        await pump();
    };

    return (
        <div className="container">
            <h1>Медицинский ассистент</h1>
            <p>Загрузите PDF-файл с клиническими рекомендациями</p>

            <input type="file" accept=".pdf" onChange={handleUpload} />

            {loading && <p>⏳ Генерирую алгоритм…</p>}

            <pre style={{ whiteSpace: "pre-wrap", marginTop: 24 }}>{text}</pre>
        </div>
    );
}

export default App;