# ============================================================
# ClinicalDistill MCP Server
# ============================================================
# Exposes one tool: extract_symptoms
# Takes clinical text → returns structured JSON
# Uses OpenAI GPT-4o for extraction
# ============================================================

from mcp.server.fastmcp import FastMCP
from openai import OpenAI
import json
import re

# Initialize MCP server and OpenAI client
mcp = FastMCP("ClinicalDistill")
client = OpenAI()  # uses OPENAI_API_KEY from environment

SYSTEM_PROMPT = """You are a clinical NLP model. Extract ONLY medical symptoms from the clinical note.

Return JSON in this exact format:
{
  "symptoms": ["symptom1", "symptom2"],
  "duration": ["duration1", "duration2"],
  "severity": ["severity1", "severity2"],
  "urgent": true/false
}

Rules:
- symptoms: ONLY medical symptoms (fever, back pain, headache, nausea, cough).
  NOT observations, context, or descriptions like "seems okay" or "a little cranky"
- duration: how long each symptom has lasted. Use "unspecified" if not mentioned
- severity: how severe each symptom is. Use "unspecified" if not clearly stated
- urgent=true ONLY for: chest pain, difficulty breathing, stroke symptoms
  (slurred speech, facial drooping, arm weakness), severe bleeding, loss of consciousness
- urgent=false for: back pain, headache, nausea, fever, diarrhea, fatigue,
  sneezing, runny nose, cough, irritability, dizziness, stomach ache
- Never duplicate symptoms
- All arrays must be the same length
- Return ONLY the JSON object, no explanation, no markdown"""


def clean_json(raw: str) -> str:
    """Strip markdown code fences if present"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


@mcp.tool()
def extract_symptoms(clinical_note: str) -> str:
    """
    Extract structured clinical symptoms from unstructured medical text.

    Converts casual or clinical patient descriptions into structured JSON
    containing symptoms, duration, severity, and an urgency flag.

    Args:
        clinical_note: Raw clinical text or patient description.
                      Can be formal clinical language or casual patient speech.

    Returns:
        JSON string with keys:
        - symptoms: list of medical symptoms found
        - duration: how long each symptom has lasted
        - severity: severity level of each symptom
        - urgent: true if immediate attention may be required

    Example:
        Input:  "chest feels weird, tired walking to kitchen for 3 days"
        Output: {
                  "symptoms": ["chest discomfort", "fatigue"],
                  "duration": ["3 days", "3 days"],
                  "severity": ["unspecified", "mild"],
                  "urgent": true
                }
    """
    if not clinical_note or not clinical_note.strip():
        return json.dumps({"error": "No clinical note provided"})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": clinical_note.strip()}
            ],
            temperature=0.1,   # low temperature for consistent structured output
            max_tokens=500
        )

        raw = response.choices[0].message.content
        raw = clean_json(raw)

        # Validate it parses as JSON
        result = json.loads(raw)

        # Ensure all required fields exist
        for key in ("symptoms", "duration", "severity"):
            if key not in result or not isinstance(result[key], list):
                result[key] = []
        if "urgent" not in result:
            result["urgent"] = False

        # Pad arrays to same length
        n = max(len(result["symptoms"]), 1)
        for key in ("symptoms", "duration", "severity"):
            while len(result[key]) < n:
                result[key].append("unspecified")

        return json.dumps(result, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({
            "error": "Failed to parse model output as JSON",
            "details": str(e),
            "raw_output": raw if 'raw' in locals() else "no output"
        })
    except Exception as e:
        return json.dumps({
            "error": "Extraction failed",
            "details": str(e)
        })


@mcp.tool()
def check_urgency(clinical_note: str) -> str:
    """
    Quickly check if a clinical note describes an urgent medical situation.

    Returns a simple urgent/non-urgent assessment with reasoning.

    Args:
        clinical_note: Raw clinical text or patient description

    Returns:
        JSON string with:
        - urgent: true/false
        - reason: brief explanation of the urgency assessment
        - recommended_action: suggested next step
    """
    if not clinical_note or not clinical_note.strip():
        return json.dumps({"error": "No clinical note provided"})

    urgency_prompt = """Assess whether this clinical note describes an urgent situation.

Return JSON:
{
  "urgent": true/false,
  "reason": "brief explanation",
  "recommended_action": "suggested next step"
}

urgent=true ONLY for: chest pain, difficulty breathing, stroke symptoms,
severe bleeding, loss of consciousness, severe allergic reaction.
urgent=false for everything else.
Return ONLY the JSON, no markdown."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": urgency_prompt},
                {"role": "user", "content": clinical_note.strip()}
            ],
            temperature=0.1,
            max_tokens=200
        )

        raw = clean_json(response.choices[0].message.content)
        result = json.loads(raw)
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "error": "Urgency check failed",
            "details": str(e)
        })


if __name__ == "__main__":
    mcp.run(transport="stdio")