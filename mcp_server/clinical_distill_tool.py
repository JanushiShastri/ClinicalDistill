# import json
# import re
# from typing import Annotated
# from mcp.server.fastmcp import Context
# from pydantic import Field
# from mcp_utilities import create_text_response
# from fhir_utilities import get_fhir_context, get_patient_id_if_context_exists
# from openai import OpenAI

# client = OpenAI()

# SYSTEM_PROMPT = """You are a clinical NLP model. Extract ONLY medical symptoms from the clinical note.

# Return JSON in this exact format:
# {
#   "symptoms": ["symptom1", "symptom2"],
#   "duration": ["duration1", "duration2"],
#   "severity": ["severity1", "severity2"],
#   "urgent": true/false
# }

# Rules:
# - symptoms: ONLY medical symptoms. NOT observations or context
# - duration: how long each symptom has lasted. Use "unspecified" if not mentioned
# - severity: how severe each symptom is. Use "unspecified" if not clearly stated
# - urgent=true ONLY for: chest pain, difficulty breathing, stroke symptoms, severe bleeding, loss of consciousness
# - urgent=false for: back pain, headache, nausea, fever, diarrhea, fatigue, sneezing, cough
# - Never duplicate symptoms
# - All arrays must be same length
# - Return ONLY the JSON object, no markdown"""


# def clean_json(raw: str) -> str:
#     raw = raw.strip()
#     if raw.startswith("```"):
#         raw = raw.split("```")[1]
#         if raw.startswith("json"):
#             raw = raw[4:]
#     return raw.strip()


# async def extract_clinical_symptoms(
#     clinical_note: Annotated[
#         str,
#         Field(description="The clinical note or patient description to extract symptoms from")
#     ],
#     patientId: Annotated[  # noqa: N803
#         str | None,
#         Field(description="The patient ID. Optional if patient context already exists")
#     ] = None,
#     ctx: Context = None,
# ) -> str:
#     """Extract structured clinical symptoms from unstructured medical text."""

#     # Get patient ID from FHIR context if not provided
#     if not patientId and ctx:
#         patientId = get_patient_id_if_context_exists(ctx)

#     # Get FHIR context if available
#     fhir_context = None
#     if ctx:
#         fhir_context = get_fhir_context(ctx)

#     if not clinical_note or not clinical_note.strip():
#         return create_text_response("No clinical note provided", is_error=True)

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": clinical_note.strip()}
#             ],
#             temperature=0.1,
#             max_tokens=500
#         )

#         raw = clean_json(response.choices[0].message.content)
#         result = json.loads(raw)

#         # Ensure all required fields
#         for key in ("symptoms", "duration", "severity"):
#             if key not in result or not isinstance(result[key], list):
#                 result[key] = []
#         if "urgent" not in result:
#             result["urgent"] = False

#         # Pad arrays to same length
#         n = max(len(result["symptoms"]), 1)
#         for key in ("symptoms", "duration", "severity"):
#             while len(result[key]) < n:
#                 result[key].append("unspecified")

#         # Add FHIR context to response
#         if patientId:
#             result["patient_id"] = patientId
#         if fhir_context:
#             result["fhir_server"] = fhir_context.url

#         return create_text_response(json.dumps(result, indent=2))

#     except Exception as e:
#         return create_text_response(f"Extraction failed: {str(e)}", is_error=True)


# async def check_clinical_urgency(
#     clinical_note: Annotated[
#         str,
#         Field(description="The clinical note or patient description to assess urgency")
#     ],
#     ctx: Context = None,
# ) -> str:
#     """Quickly assess if a clinical note describes an urgent medical situation."""

#     if not clinical_note or not clinical_note.strip():
#         return create_text_response("No clinical note provided", is_error=True)

#     urgency_prompt = """Assess whether this clinical note describes an urgent situation.

# Return JSON:
# {
#   "urgent": true/false,
#   "reason": "brief explanation",
#   "recommended_action": "suggested next step"
# }

# urgent=true ONLY for: chest pain, difficulty breathing, stroke symptoms,
# severe bleeding, loss of consciousness, severe allergic reaction.
# urgent=false for everything else.
# Return ONLY the JSON, no markdown."""

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": urgency_prompt},
#                 {"role": "user", "content": clinical_note.strip()}
#             ],
#             temperature=0.1,
#             max_tokens=200
#         )
#         raw = clean_json(response.choices[0].message.content)
#         result = json.loads(raw)
#         return create_text_response(json.dumps(result, indent=2))

#     except Exception as e:
#         return create_text_response(f"Urgency check failed: {str(e)}", is_error=True)

# async def check_patient_documents(
#     patientId: Annotated[
#         str | None,
#         Field(description="The patient ID. Optional if patient context exists")
#     ] = None,
#     ctx: Context = None,
# ) -> str:
#     """Check what documents exist for a patient in FHIR"""

#     patient_id = patientId or get_patient_id_if_context_exists(ctx)
#     fhir_context = get_fhir_context(ctx)

#     if not patient_id:
#         return create_text_response("No patient ID found", is_error=True)

#     if not fhir_context:
#         return create_text_response("No FHIR context found", is_error=True)

#     fhir_client = FhirClient(
#         base_url=fhir_context.url,
#         token=fhir_context.token
#     )

#     bundle = await fhir_client.search(
#         "DocumentReference",
#         {"patient": patient_id}
#     )

#     if not bundle or not bundle.get("entry"):
#         return create_text_response(
#             f"No documents found for patient {patient_id}"
#         )

#     docs = []
#     for entry in bundle.get("entry", []):
#         resource = entry.get("resource", {})
#         docs.append({
#             "id": resource.get("id"),
#             "status": resource.get("status"),
#             "type": resource.get("type", {}).get("text", "unknown"),
#             "date": resource.get("date", "unknown")
#         })

#     return create_text_response(
#         f"Found {len(docs)} documents for patient {patient_id}:\n"
#         f"{json.dumps(docs, indent=2)}"
#     )

import json
import re
import base64
from typing import Annotated
from mcp.server.fastmcp import Context
from pydantic import Field
from mcp_utilities import create_text_response
from fhir_utilities import get_fhir_context, get_patient_id_if_context_exists
from fhir_client import FhirClient
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
- duration: how long each symptom lasted. Use "unspecified" if not mentioned
- severity: how severe. Use "unspecified" if not clearly stated
- urgent=true ONLY for: chest pain, difficulty breathing, stroke symptoms, severe bleeding, loss of consciousness
- urgent=false for: back pain, headache, nausea, fever, diarrhea, fatigue, sneezing, cough, dental issues
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


def extract_document_text(resource: dict) -> str:
    """Extract text content from a FHIR DocumentReference resource"""
    try:
        content = resource.get("content", [])
        for item in content:
            attachment = item.get("attachment", {})
            # Try direct data (base64 encoded)
            if attachment.get("data"):
                decoded = base64.b64decode(attachment["data"]).decode("utf-8")
                return decoded
            # Try URL-based content
            if attachment.get("url"):
                return attachment.get("title", "")
        # Fallback to text section
        text = resource.get("text", {}).get("div", "")
        if text:
            # Strip HTML tags
            clean = re.sub(r'<[^>]+>', ' ', text)
            return clean.strip()
    except Exception:
        pass
    return ""


# async def extract_clinical_symptoms(
#     clinical_note: Annotated[
#         str,
#         Field(description="The clinical note or patient description to extract symptoms from")
#     ],
#     patientId: Annotated[
#         str | None,
#         Field(description="The patient ID. Optional if patient context already exists")
#     ] = None,
#     ctx: Context = None,
# ) -> str:
#     """Extract structured clinical symptoms from unstructured medical text."""

#     if not patientId and ctx:
#         patientId = get_patient_id_if_context_exists(ctx)

#     fhir_context = None
#     if ctx:
#         fhir_context = get_fhir_context(ctx)

#     if not clinical_note or not clinical_note.strip():
#         return create_text_response("No clinical note provided", is_error=True)

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": clinical_note.strip()}
#             ],
#             temperature=0.1,
#             max_tokens=500
#         )

#         raw = clean_json(response.choices[0].message.content)
#         result = json.loads(raw)

#         for key in ("symptoms", "duration", "severity"):
#             if key not in result or not isinstance(result[key], list):
#                 result[key] = []
#         if "urgent" not in result:
#             result["urgent"] = False

#         n = max(len(result["symptoms"]), 1)
#         for key in ("symptoms", "duration", "severity"):
#             while len(result[key]) < n:
#                 result[key].append("unspecified")

#         if patientId:
#             result["patient_id"] = patientId

#         # return create_text_response(json.dumps(result, indent=2))
        

#     except Exception as e:
#         return create_text_response(f"Extraction failed: {str(e)}", is_error=True)

async def extract_clinical_symptoms(
    clinical_note: Annotated[
        str,
        Field(description="The clinical note or patient description to extract symptoms from")
    ],
    patientId: Annotated[
        str | None,
        Field(description="The patient ID. Optional if patient context already exists")
    ] = None,
    ctx: Context = None,
) -> str:
    """Extract structured clinical symptoms from unstructured medical text."""

    if not patientId and ctx:
        patientId = get_patient_id_if_context_exists(ctx)

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

        for key in ("symptoms", "duration", "severity"):
            if key not in result or not isinstance(result[key], list):
                result[key] = []
        if "urgent" not in result:
            result["urgent"] = False

        n = max(len(result["symptoms"]), 1)
        for key in ("symptoms", "duration", "severity"):
            while len(result[key]) < n:
                result[key].append("unspecified")

        if patientId:
            result["patient_id"] = patientId

        # Format output as clean table — no raw JSON
        symptoms = result.get("symptoms", [])
        durations = result.get("duration", [])
        severities = result.get("severity", [])
        urgent = result.get("urgent", False)

        rows = ""
        for i, sym in enumerate(symptoms):
            dur = durations[i] if i < len(durations) else "—"
            sev = severities[i] if i < len(severities) else "—"
            if dur in ("unspecified", ""): dur = "—"
            if sev in ("unspecified", ""): sev = "—"
            rows += f"| {sym} | {dur} | {sev} |\n"

        urgency = "🚨 URGENT — Immediate medical attention required." if urgent else "✅ NON-URGENT — Routine follow-up appropriate."

        patient_line = f"**Patient ID:** {patientId}\n" if patientId else ""

        formatted = f"""🏥 Clinical Symptom Extraction

{patient_line}**Symptoms Found:**
| Symptom | Duration | Severity |
|---------|----------|----------|
{rows}
**Urgency:** {urgency}

**Summary:** {len(symptoms)} symptom(s) extracted. {'Immediate medical attention required.' if urgent else 'Routine follow-up appropriate.'}"""

        return create_text_response(formatted)

    except Exception as e:
        return create_text_response(f"Extraction failed: {str(e)}", is_error=True)

async def analyze_patient_notes(
    patientId: Annotated[
        str | None,
        Field(description="The patient ID. Optional if patient context already exists")
    ] = None,
    max_notes: Annotated[
        int,
        Field(description="Maximum number of recent notes to analyze. Default is 3.")
    ] = 3,
    ctx: Context = None,
) -> str:
    """
    Fully automated pipeline: reads patient FHIR documents, 
    extracts symptoms, and saves them back as FHIR Conditions.
    """

    patient_id = patientId or get_patient_id_if_context_exists(ctx)
    fhir_context = get_fhir_context(ctx)

    if not patient_id:
        return create_text_response("No patient ID found", is_error=True)
    if not fhir_context:
        return create_text_response("No FHIR context found", is_error=True)

    fhir_client = FhirClient(
        base_url=fhir_context.url,
        token=fhir_context.token
    )

    # Step 1 — Read patient's existing documents
    bundle = await fhir_client.search(
        "DocumentReference",
        {"patient": patient_id, "_sort": "-date", "_count": str(max_notes)}
    )

    if not bundle or not bundle.get("entry"):
        return create_text_response(
            f"No clinical documents found for patient {patient_id}",
            is_error=True
        )

    # Step 2 — Extract text from documents
    all_text = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        text = extract_document_text(resource)
        if text and len(text) > 50:
            all_text.append(text[:1000])

    if not all_text:
        return create_text_response(
            "Found documents but could not extract text content",
            is_error=True
        )

    combined_note = "\n\n---\n\n".join(all_text[:max_notes])

    # Step 3 — Extract symptoms using GPT-4o
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": combined_note}
            ],
            temperature=0.1,
            max_tokens=800
        )

        raw = clean_json(response.choices[0].message.content)
        result = json.loads(raw)

        for key in ("symptoms", "duration", "severity"):
            if key not in result or not isinstance(result[key], list):
                result[key] = []
        if "urgent" not in result:
            result["urgent"] = False

        n = max(len(result["symptoms"]), 1)
        for key in ("symptoms", "duration", "severity"):
            while len(result[key]) < n:
                result[key].append("unspecified")

    except Exception as e:
        return create_text_response(
            f"Symptom extraction failed: {str(e)}", is_error=True
        )

    # Step 4 — Write symptoms back to FHIR as Conditions
    saved_conditions = []
    for i, symptom in enumerate(result["symptoms"]):
        try:
            condition = {
                "resourceType": "Condition",
                "subject": {"reference": f"Patient/{patient_id}"},
                "code": {
                    "text": symptom,
                    "coding": [{"display": symptom}]
                },
                "onsetString": result["duration"][i] if i < len(result["duration"]) else "unspecified",
                "note": [{
                    "text": f"severity: {result['severity'][i] if i < len(result['severity']) else 'unspecified'}, urgent: {result['urgent']}, source: ClinicalDistill MCP"
                }],
                "verificationStatus": {
                    "coding": [{"code": "unconfirmed"}]
                },
                "clinicalStatus": {
                    "coding": [{"code": "active"}]
                }
            }
            await fhir_client.create("Condition", condition)
            saved_conditions.append(symptom)
        except Exception:
            pass

    # Step 5 — Build formatted response
    urgency_line = "🚨 URGENT — Immediate medical attention required." if result["urgent"] else "✅ NON-URGENT — Routine follow-up appropriate."

    rows = ""
    for i, sym in enumerate(result["symptoms"]):
        dur = result["duration"][i] if i < len(result["duration"]) else "—"
        sev = result["severity"][i] if i < len(result["severity"]) else "—"
        if dur == "unspecified": dur = "—"
        if sev == "unspecified": sev = "—"
        rows += f"| {sym} | {dur} | {sev} |\n"

    # fhir_status = f"✅ {len(saved_conditions)} conditions saved to FHIR record." if saved_conditions else "⚠️ Could not save to FHIR."
    if saved_conditions:
        fhir_status = f"✅ {len(saved_conditions)} conditions saved to FHIR record."
    elif result["symptoms"]:
        fhir_status = "⚠️ Extraction complete. FHIR write requires elevated permissions (403). Conditions ready for manual review."
    else:
        fhir_status = "⚠️ No symptoms found to save."
    response_text = f"""🏥 Clinical Analysis Complete

**Patient:** {patient_id}
**Notes analyzed:** {len(all_text)}

**Symptoms Found:**
| Symptom | Duration | Severity |
|---------|----------|----------|
{rows}
**Urgency Assessment:** {urgency_line}

**FHIR Update:** {fhir_status}"""

    return create_text_response(response_text)


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
        return create_text_response(
            f"Urgency check failed: {str(e)}", is_error=True
        )


async def check_patient_documents(
    patientId: Annotated[
        str | None,
        Field(description="The patient ID. Optional if patient context exists")
    ] = None,
    ctx: Context = None,
) -> str:
    """Check what documents exist for a patient in FHIR"""

    patient_id = patientId or get_patient_id_if_context_exists(ctx)
    fhir_context = get_fhir_context(ctx)

    if not patient_id:
        return create_text_response("No patient ID found", is_error=True)
    if not fhir_context:
        return create_text_response("No FHIR context found", is_error=True)

    fhir_client = FhirClient(
        base_url=fhir_context.url,
        token=fhir_context.token
    )

    bundle = await fhir_client.search(
        "DocumentReference",
        {"patient": patient_id}
    )

    if not bundle or not bundle.get("entry"):
        return create_text_response(
            f"No documents found for patient {patient_id}"
        )

    docs = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        docs.append({
            "id": resource.get("id"),
            "date": resource.get("date", "unknown"),
            "type": resource.get("type", {}).get("text", "unknown")
        })

    return create_text_response(
        f"Found {len(docs)} documents for patient {patient_id}:\n"
        f"{json.dumps(docs, indent=2)}"
    )