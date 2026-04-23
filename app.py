import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import re

MODEL_ID = "google/gemma-3-1b-it"
LORA_WEIGHTS = "Janushi/ClinicalDistill-Gemma-1B"

tokenizer = None
model = None

INSTRUCTION = """Extract symptoms from the clinical note and return JSON with this exact format:
{
  "symptoms": ["symptom1", "symptom2"],
  "duration": ["duration1", "duration2"],
  "severity": ["severity1", "severity2"],
  "urgent": true/false
}"""

EXAMPLES = [
    ["Patient presents with crushing chest pain radiating to left arm for 2 hours, diaphoresis, and mild shortness of breath."],
    ["Pt c/o bad headache since yesterday, some nausea, light bothers eyes. No fever."],
    ["78yo male with sudden onset severe dyspnea, orthopnea, bilateral leg swelling x 3 days, productive cough with pink frothy sputum."],
    ["Patient complains of stomach ache and diarrhea for 2 days, no blood in stool, mild cramping."],
]

CSS = """
#title { text-align: center; margin-bottom: 0.5rem; }
#subtitle { text-align: center; color: #6b7280; margin-bottom: 1.5rem; font-size: 0.95rem; }
#stats-row { display: flex; gap: 1rem; justify-content: center; margin-bottom: 1.5rem; flex-wrap: wrap; }
.stat-card {
    background: linear-gradient(135deg, #667eea20, #764ba220);
    border: 1px solid #667eea40;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    text-align: center;
    min-width: 110px;
}
.stat-val { font-size: 1.4rem; font-weight: 700; color: #4f46e5; }
.stat-lbl { font-size: 0.75rem; color: #6b7280; }
#submit-btn {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important;
    font-size: 1rem !important;
    padding: 0.75rem !important;
}
#urgent-badge-yes {
    background: #fef2f2; border: 1px solid #fca5a5;
    color: #dc2626; border-radius: 8px; padding: 0.5rem 1rem;
    font-weight: 600; text-align: center; margin-top: 0.5rem;
}
#urgent-badge-no {
    background: #f0fdf4; border: 1px solid #86efac;
    color: #16a34a; border-radius: 8px; padding: 0.5rem 1rem;
    font-weight: 600; text-align: center; margin-top: 0.5rem;
}
.symptom-table th { background: #f1f5f9 !important; }
footer { display: none !important; }
"""


def load_model():
    global tokenizer, model
    if model is not None:
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    try:
        model = PeftModel.from_pretrained(base, LORA_WEIGHTS)
        model = model.merge_and_unload()
    except Exception:
        model = base
    model.eval()


def build_prompt(clinical_note: str) -> str:
    return (
        f"<instruction>\n{INSTRUCTION}\n</instruction>\n\n"
        f"<input>\n{clinical_note.strip()}\n</input>\n\n"
        f"<output>\n"
    )


def parse_output(raw: str) -> dict:
    raw = raw.split("</output>")[0].strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    result = json.loads(raw)
    for key in ("symptoms", "duration", "severity"):
        if key not in result or not isinstance(result[key], list):
            result[key] = []
    if "urgent" not in result:
        result["urgent"] = False
    n = max(len(result["symptoms"]), len(result["duration"]), len(result["severity"]), 1)
    for key in ("symptoms", "duration", "severity"):
        while len(result[key]) < n:
            result[key].append("unspecified")
    return result


def severity_badge(s: str) -> str:
    s = s.lower()
    if any(w in s for w in ("severe", "critical", "extreme", "crushing", "sudden")):
        return f'<span style="color:#dc2626;font-weight:600">⬆ {s}</span>'
    if any(w in s for w in ("moderate", "significant")):
        return f'<span style="color:#d97706;font-weight:600">➡ {s}</span>'
    if any(w in s for w in ("mild", "slight", "minor")):
        return f'<span style="color:#16a34a;font-weight:600">⬇ {s}</span>'
    return f'<span style="color:#6b7280">{s}</span>'


def format_results(result: dict):
    symptoms = result["symptoms"]
    durations = result["duration"]
    severities = result["severity"]
    urgent = result.get("urgent", False)

    rows = ""
    for i, sym in enumerate(symptoms):
        dur = durations[i] if i < len(durations) else "unspecified"
        sev = severities[i] if i < len(severities) else "unspecified"
        bg = "#fff7f7" if urgent and i == 0 else "white"
        rows += f"""
        <tr style="background:{bg}">
            <td style="padding:10px 14px;font-weight:500">{sym}</td>
            <td style="padding:10px 14px;color:#4b5563">{dur}</td>
            <td style="padding:10px 14px">{severity_badge(sev)}</td>
        </tr>"""

    urgent_html = (
        '<div id="urgent-badge-yes">🚨 URGENT — Immediate attention may be required</div>'
        if urgent else
        '<div id="urgent-badge-no">✅ NON-URGENT — Routine follow-up appropriate</div>'
    )

    table_html = f"""
    <div style="font-family:sans-serif">
        <table style="width:100%;border-collapse:collapse;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px #0001">
            <thead>
                <tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0">
                    <th style="padding:10px 14px;text-align:left;color:#374151">Symptom</th>
                    <th style="padding:10px 14px;text-align:left;color:#374151">Duration</th>
                    <th style="padding:10px 14px;text-align:left;color:#374151">Severity</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        {urgent_html}
    </div>
    """
    return table_html


def extract(clinical_note: str):
    if not clinical_note.strip():
        return "<p style='color:#9ca3af'>Enter a clinical note to see results.</p>", "{}"

    load_model()

    prompt = build_prompt(clinical_note)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    try:
        result = parse_output(generated)
        table_html = format_results(result)
        json_out = json.dumps(result, indent=2)
        return table_html, json_out
    except (json.JSONDecodeError, KeyError, IndexError):
        err = f"<pre style='color:#dc2626'>Parse error. Raw output:\n{generated}</pre>"
        return err, "{}"


with gr.Blocks(css=CSS, title="ClinicalDistill") as demo:

    gr.HTML("""
    <h1 id="title" style="font-size:2rem;font-weight:800;background:linear-gradient(135deg,#667eea,#764ba2);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-top:1rem">
        🏥 ClinicalDistill
    </h1>
    <p id="subtitle">Structured symptom extraction from clinical notes · Gemma-3-1B fine-tuned with LoRA</p>
    <div id="stats-row">
        <div class="stat-card"><div class="stat-val">0.781</div><div class="stat-lbl">F1 Score</div></div>
        <div class="stat-card"><div class="stat-val">85.7%</div><div class="stat-lbl">Urgent Accuracy</div></div>
        <div class="stat-card"><div class="stat-val">100%</div><div class="stat-lbl">Valid JSON</div></div>
        <div class="stat-card"><div class="stat-val">1B</div><div class="stat-lbl">Parameters</div></div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            note_input = gr.Textbox(
                label="Clinical Note",
                placeholder="e.g. Patient presents with crushing chest pain for 2 hours, diaphoresis...",
                lines=7,
                show_label=True,
            )
            submit_btn = gr.Button("⚡ Extract Symptoms", variant="primary", elem_id="submit-btn")

            gr.Examples(
                examples=EXAMPLES,
                inputs=note_input,
                label="Try an Example",
            )

        with gr.Column(scale=1):
            table_output = gr.HTML(
                label="Extracted Symptoms",
                value="<p style='color:#9ca3af;text-align:center;margin-top:2rem'>Results will appear here.</p>",
            )
            with gr.Accordion("Raw JSON Output", open=False):
                json_output = gr.Code(language="json", label="")

    submit_btn.click(fn=extract, inputs=note_input, outputs=[table_output, json_output])
    note_input.submit(fn=extract, inputs=note_input, outputs=[table_output, json_output])

    gr.HTML("""
    <div style="text-align:center;margin-top:2rem;padding-top:1rem;
        border-top:1px solid #e5e7eb;color:#9ca3af;font-size:0.85rem">
        ClinicalDistill · Fine-tuned on 145 synthetic clinical examples (cardiac, respiratory, neurological, GI)
        · <a href="https://github.com/Janushi/ClinicalDistill" style="color:#667eea">GitHub</a>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
