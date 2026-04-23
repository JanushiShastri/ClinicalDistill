import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re

MODEL_ID = "Janushi/ClinicalDistill-Gemma-1B"

tokenizer = None
model = None

INSTRUCTION = """You are a clinical NLP model. Extract ONLY medical symptoms from the clinical note.

Return JSON in this exact format:
{
  "symptoms": ["symptom1", "symptom2"],
  "duration": ["duration1", "duration2"],
  "severity": ["severity1", "severity2"],
  "urgent": true/false
}

Rules:
- symptoms: ONLY medical symptoms (fever, back pain, headache, nausea, cough, dizziness). NOT observations, context, or descriptions like "seems okay", "a little cranky", "not sure"
- duration: how long each symptom has lasted. Use "unspecified" if not mentioned
- severity: how severe each symptom is. Use "unspecified" if not clearly stated — do NOT guess severity
- urgent=true ONLY for: chest pain, difficulty breathing, stroke symptoms (slurred speech, facial drooping, arm weakness), severe bleeding, loss of consciousness
- urgent=false for: back pain, headache, nausea, fever, diarrhea, fatigue, sneezing, runny nose, cough, irritability, dizziness, stomach ache
- Never duplicate symptoms
- All arrays must be the same length"""

EXAMPLES = [
    ["been feeling off for a few days, chest feels weird and i get tired just walking to the kitchen"],
    ["my back's been killing me since last week, hurts way more when i sit, not sure if i pulled something"],
    ["crushing chest pain radiating to jaw for 30 mins, sweating, feels like something is very wrong"],
    ["kid woke up hot last night, been sneezing a lot, seems okay otherwise just a little cranky"],
    ["stomach's been acting up since yesterday, went to the bathroom like 4 times, feeling really drained"],
    ["these headaches keep coming back, nothing crazy but annoying, sometimes feel dizzy too"],
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
footer { display: none !important; }
"""

# Phrases that are NOT valid medical symptoms
NON_SYMPTOM_PHRASES = [
    "seems okay", "seems fine", "otherwise fine", "no fever", "a little",
    "otherwise", "seems", "appears", "looks", "none", "normal", "okay",
    "fine", "not sure", "cranky", "irritable", "fussy", "acting up",
    "feeling off", "feeling drained", "feeling tired", "feeling weak"
]

def is_valid_symptom(s: str) -> bool:
    s_lower = s.lower().strip()
    # Too long to be a real symptom (more than 5 words)
    if len(s_lower.split()) > 5:
        return False
    # Contains non-symptom phrases
    if any(phrase in s_lower for phrase in NON_SYMPTOM_PHRASES):
        return False
    # Too short to be meaningful
    if len(s_lower) < 3:
        return False
    return True


def load_model():
    global tokenizer, model
    if model is not None:
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,  # float32 for CPU
        device_map="cpu",
    )
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

    # Deduplicate symptoms preserving order
    seen = set()
    unique_indices = []
    for i, sym in enumerate(result["symptoms"]):
        sym_lower = sym.lower().strip()
        if sym_lower not in seen:
            seen.add(sym_lower)
            unique_indices.append(i)

    result["symptoms"] = [result["symptoms"][i] for i in unique_indices]
    result["duration"] = [
        result["duration"][i] if i < len(result["duration"]) else "unspecified"
        for i in unique_indices
    ]
    result["severity"] = [
        result["severity"][i] if i < len(result["severity"]) else "unspecified"
        for i in unique_indices
    ]

    # Filter out non-medical symptom descriptions
    valid_indices = [
        i for i, sym in enumerate(result["symptoms"])
        if is_valid_symptom(sym)
    ]
    # Keep at least one symptom even if filter removes everything
    if not valid_indices and result["symptoms"]:
        valid_indices = [0]

    result["symptoms"] = [result["symptoms"][i] for i in valid_indices]
    result["duration"] = [result["duration"][i] for i in valid_indices]
    result["severity"] = [result["severity"][i] for i in valid_indices]

    # Pad arrays to same length
    n = len(result["symptoms"]) or 1
    for key in ("symptoms", "duration", "severity"):
        while len(result[key]) < n:
            result[key].append("unspecified")

    return result


def severity_badge(s: str) -> str:
    s = (s or "").lower().strip()
    if not s or s in ("unspecified", "unknown", "n/a", "none", "not mentioned", "—", "not stated"):
        return '<span style="color:#d1d5db;font-style:italic">—</span>'
    if any(w in s for w in ("severe", "critical", "extreme", "crushing", "sudden", "acute", "high", "sharp")):
        return f'<span style="background:#fef2f2;color:#dc2626;font-weight:600;padding:2px 8px;border-radius:4px;font-size:0.85rem">▲ {s}</span>'
    if any(w in s for w in ("moderate", "significant", "worsening", "progressive", "persistent")):
        return f'<span style="background:#fffbeb;color:#d97706;font-weight:600;padding:2px 8px;border-radius:4px;font-size:0.85rem">● {s}</span>'
    if any(w in s for w in ("mild", "slight", "minor", "low", "minimal", "light")):
        return f'<span style="background:#f0fdf4;color:#16a34a;font-weight:600;padding:2px 8px;border-radius:4px;font-size:0.85rem">▼ {s}</span>'
    return f'<span style="background:#f1f5f9;color:#475569;padding:2px 8px;border-radius:4px;font-size:0.85rem">{s}</span>'


def format_duration(d: str) -> str:
    d = (d or "").lower().strip()
    if not d or d in ("unspecified", "unknown", "n/a", "none", "not mentioned", "not stated"):
        return '<span style="color:#d1d5db;font-style:italic">—</span>'
    return f'<span style="color:#4b5563">{d}</span>'


def format_results(result: dict) -> str:
    symptoms = result["symptoms"]
    durations = result["duration"]
    severities = result["severity"]
    urgent = result.get("urgent", False)
    accent = "#dc2626" if urgent else "#667eea"

    rows = ""
    for i, sym in enumerate(symptoms):
        dur = durations[i] if i < len(durations) else "unspecified"
        sev = severities[i] if i < len(severities) else "unspecified"
        bg = "#fff7f7" if urgent else ("#f8faff" if i % 2 == 0 else "white")

        rows += f"""
        <tr style="background:{bg}">
            <td style="padding:10px 14px;font-weight:600;color:#111827;
                border-left:3px solid {accent}">{sym}</td>
            <td style="padding:10px 14px">{format_duration(dur)}</td>
            <td style="padding:10px 14px">{severity_badge(sev)}</td>
        </tr>"""

    urgent_html = (
        '<div id="urgent-badge-yes">🚨 URGENT — Immediate attention may be required</div>'
        if urgent else
        '<div id="urgent-badge-no">✅ NON-URGENT — Routine follow-up appropriate</div>'
    )

    return f"""
    <div style="font-family:sans-serif">
        <table style="width:100%;border-collapse:collapse;border-radius:10px;
            overflow:hidden;box-shadow:0 1px 4px #0001">
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


def extract(clinical_note: str, state: dict):
    if not clinical_note.strip():
        return (
            "<p style='color:#9ca3af'>Enter a clinical note to see results.</p>",
            "{}",
            state,
            gr.update(visible=True),  # keep warning visible
        )

    load_model()
    prompt = build_prompt(clinical_note)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
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
        new_state = {"table": table_html, "json": json_out}
        return table_html, json_out, new_state, gr.update(visible=False)  # hide warning
    except (json.JSONDecodeError, KeyError, IndexError):
        err = f"<pre style='color:#dc2626'>Parse error. Raw output:\n{generated}</pre>"
        return err, "{}", state, gr.update(visible=False)


with gr.Blocks(css=CSS, title="ClinicalDistill") as demo:

    result_state = gr.State(value={"table": "", "json": "{}"})

    gr.HTML("""
    <h1 id="title" style="font-size:2rem;font-weight:800;
        background:linear-gradient(135deg,#667eea,#764ba2);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        margin-top:1rem">
        🏥 ClinicalDistill
    </h1>
    <p id="subtitle">
        Structured symptom extraction from clinical notes ·
        Gemma-3-1B fine-tuned with LoRA
    </p>
    <div id="stats-row">
        <div class="stat-card">
            <div class="stat-val">0.781</div>
            <div class="stat-lbl">F1 Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">85.7%</div>
            <div class="stat-lbl">Urgent Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">100%</div>
            <div class="stat-lbl">Valid JSON</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">1B</div>
            <div class="stat-lbl">Parameters</div>
        </div>
    </div>
    """)

    # Warning banner — hidden after first inference
    cpu_warning = gr.HTML(
        value="""
        <div style="text-align:center;margin-bottom:1rem;padding:0.6rem 1rem;
            background:#fffbeb;border:1px solid #fde68a;border-radius:8px;
            color:#92400e;font-size:0.85rem;max-width:600px;
            margin-left:auto;margin-right:auto">
            ⏳ Running on CPU — inference takes ~60 seconds.
            Results persist after completion.
        </div>
        """,
        visible=True,
    )

    with gr.Row():
        with gr.Column(scale=1):
            note_input = gr.Textbox(
                label="Clinical Note",
                placeholder="e.g. been feeling off, chest feels weird and i get tired walking around...",
                lines=7,
            )
            submit_btn = gr.Button(
                "⚡ Extract Symptoms",
                variant="primary",
                elem_id="submit-btn",
            )
            gr.Examples(
                examples=EXAMPLES,
                inputs=note_input,
                label="Try an Example",
            )

        with gr.Column(scale=1):
            table_output = gr.HTML(
                value="<p style='color:#9ca3af;text-align:center;margin-top:2rem'>"
                      "Results will appear here.</p>",
            )
            with gr.Accordion("Raw JSON Output", open=False):
                json_output = gr.Code(language="json", label="")

    submit_btn.click(
        fn=extract,
        inputs=[note_input, result_state],
        outputs=[table_output, json_output, result_state, cpu_warning],
    )
    note_input.submit(
        fn=extract,
        inputs=[note_input, result_state],
        outputs=[table_output, json_output, result_state, cpu_warning],
    )

    gr.HTML("""
    <div style="text-align:center;margin-top:2rem;padding-top:1rem;
        border-top:1px solid #e5e7eb;color:#9ca3af;font-size:0.85rem">
        ClinicalDistill · Fine-tuned on 145 synthetic clinical examples
        (cardiac, respiratory, neurological, GI) ·
        <a href="https://github.com/JanushiShastri/ClinicalDistill"
           style="color:#667eea">GitHub</a>
    </div>
    """)

demo.launch(ssr_mode=False)