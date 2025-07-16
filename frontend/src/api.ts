export async function generateAlgorithm(
    file: File,
    onChunk: (chunk: string) => void
): Promise<void> {
    const form = new FormData();
    form.append("pdf", file);

    const res = await fetch("http://localhost:8000/generate", {
        method: "POST",
        body: form,
    });

    if (!res.ok) throw new Error(await res.text());

    const reader = res.body!.getReader();
    const decoder = new TextDecoder("utf-8");
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const lines = decoder.decode(value).split("\n");
        for (const line of lines) {
            if (line.startsWith("data: ")) {
                const chunk = line.slice(6);
                if (chunk === "[DONE]") return;
                onChunk(chunk);
            }
        }
    }
}