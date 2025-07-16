import React, { useState, useRef } from "react";
import { generateAlgorithm } from "./api";

type Message = { role: "user" | "bot"; text: string };

function App() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [loading, setLoading] = useState(false);
    const fileRef = useRef<HTMLInputElement>(null);

    const handleUpload = async () => {
        const file = fileRef.current?.files?.[0];
        if (!file) return;
        setLoading(true);
        setMessages([{ role: "user", text: `📄 ${file.name}` }]);
        try {
            await generateAlgorithm(file, (chunk) => {
                setMessages((prev) => {
                    const last = prev[prev.length - 1];
                    if (last?.role === "bot") {
                        return [
                            ...prev.slice(0, -1),
                            { role: "bot", text: last.text + chunk },
                        ];
                    }
                    return [...prev, { role: "bot", text: chunk }];
                });
            });
        } catch (e) {
            setMessages((prev) => [
                ...prev,
                { role: "bot", text: "❌ Ошибка: " + (e as Error).message },
            ]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ maxWidth: 800, margin: "0 auto", padding: 24 }}>
            <h2>Медицинский ассистент</h2>
            <input type="file" accept=".pdf" ref={fileRef} />
            <button onClick={handleUpload} disabled={loading}>
                {loading ? "Генерирую…" : "Сгенерировать алгоритм"}
            </button>

            <div style={{ marginTop: 24, whiteSpace: "pre-wrap" }}>
                {messages.map((m, i) => (
                    <div key={i} style={{ marginBottom: 8 }}>
                        <strong>{m.role === "user" ? "Вы: " : "ИИ: "}</strong>
                        {m.text}
                    </div>
                ))}
            </div>
        </div>
    );
}

export default App;