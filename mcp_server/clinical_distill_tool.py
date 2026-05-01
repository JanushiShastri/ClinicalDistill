import json
import re
from typing import Annotated
from mcp.server.fastmcp import Context
from pydantic import Field
from mcp_utilities import create_text_response
from fhir_utilities import get_fhir_context, get_patient_id_if_context_exists
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """You are a clinical NLP model. Extract ONLY medical symptoms from the clinical note.

Return JSON in this exact format:
{
  "symptoms": ["symptom1", "symptom2"],
  "duration": ["duration1", "duration2"],
  "severity": ["severity1", "severity2"],
  "urgent": true/false
}

Rules:
- symptoms: ONLY medical symptoms. NOT observations or context
- duration: how long each symptom has lasted. Use "unspecified" if not mentioned
- severity: how severe each symptom is. Use "unspecified" if not clearly stated
- urgent=true ONLY for: chest pain, difficulty breathing, stroke symptoms, severe bleeding, loss of consciousness
- urgent=false for: back pain, headache, nausea, fever, diarrhea, fatigue, sneezing, cough
- Never duplicate symptoms
- All arrays must be same length
- Return ONLY the JSON object, no markdown"""


def clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


async def extract_clinical_symptoms(
    clinical_note: Annotated[
        str,
        Field(description="The clinical note or patient description to extract symptoms from")
    ],
    patientId: Annotated[  # noqa: N803
        str | None,
        Field(description="The patient ID. Optional if patient context already exists")
    ] = None,
    ctx: Context = None,
) -> str:
    """Extract structured clinical symptoms from unstructured medical text."""

    # Get patient ID from FHIR context if not provided
    if not patientId and ctx:
        patientId = get_patient_id_if_context_exists(ctx)

    # Get FHIR context if available
    fhir_context = None
    if ctx:
        fhir_context = get_fhir_context(ctx)

    if not clinical_note or not clinical_note.strip():
        return create_text_response("No clinical note provided", is_error=True)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": clinical_note.strip()}
            ],
            temperature=0.1,
            max_tokens=500
        )

        raw = clean_json(response.choices[0].message.content)
        result = json.loads(raw)

        # Ensure all required fields
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

        # Add FHIR context to response
        if patientId:
            result["patient_id"] = patientId
        if fhir_context:
            result["fhir_server"] = fhir_context.url

        return create_text_response(json.dumps(result, indent=2))

    except Exception as e:
        return create_text_response(f"Extraction failed: {str(e)}", is_error=True)


async def check_clinical_urgency(
    clinical_note: Annotated[
        str,
        Field(description="The clinical note or patient description to assess urgency")
    ],
    ctx: Context = None,
) -> str:
    """Quickly assess if a clinical note describes an urgent medical situation."""

    if not clinical_note or not clinical_note.strip():
        return create_text_response("No clinical note provided", is_error=True)

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
        return create_text_response(json.dumps(result, indent=2))

    except Exception as e:
        return create_text_response(f"Urgency check failed: {str(e)}", is_error=True)
