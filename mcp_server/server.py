# # ============================================================
# # ClinicalDistill MCP Server
# # ============================================================
# # Exposes one tool: extract_symptoms
# # Takes clinical text → returns structured JSON
# # Uses OpenAI GPT-4o for extraction
# # ============================================================

# from mcp.server.fastmcp import FastMCP
# from openai import OpenAI
# import json
# import re

# # Initialize MCP server and OpenAI client
# mcp = FastMCP("ClinicalDistill")
# client = OpenAI()  # uses OPENAI_API_KEY from environment

# SYSTEM_PROMPT = """You are a clinical NLP model. Extract ONLY medical symptoms from the clinical note.

# Return JSON in this exact format:
# {
#   "symptoms": ["symptom1", "symptom2"],
#   "duration": ["duration1", "duration2"],
#   "severity": ["severity1", "severity2"],
#   "urgent": true/false
# }

# Rules:
# - symptoms: ONLY medical symptoms (fever, back pain, headache, nausea, cough).
#   NOT observations, context, or descriptions like "seems okay" or "a little cranky"
# - duration: how long each symptom has lasted. Use "unspecified" if not mentioned
# - severity: how severe each symptom is. Use "unspecified" if not clearly stated
# - urgent=true ONLY for: chest pain, difficulty breathing, stroke symptoms
#   (slurred speech, facial drooping, arm weakness), severe bleeding, loss of consciousness
# - urgent=false for: back pain, headache, nausea, fever, diarrhea, fatigue,
#   sneezing, runny nose, cough, irritability, dizziness, stomach ache
# - Never duplicate symptoms
# - All arrays must be the same length
# - Return ONLY the JSON object, no explanation, no markdown"""


# def clean_json(raw: str) -> str:
#     """Strip markdown code fences if present"""
#     raw = raw.strip()
#     if raw.startswith("```"):
#         raw = raw.split("```")[1]
#         if raw.startswith("json"):
#             raw = raw[4:]
#     return raw.strip()


# @mcp.tool()
# def extract_symptoms(clinical_note: str) -> str:
#     """
#     Extract structured clinical symptoms from unstructured medical text.

#     Converts casual or clinical patient descriptions into structured JSON
#     containing symptoms, duration, severity, and an urgency flag.

#     Args:
#         clinical_note: Raw clinical text or patient description.
#                       Can be formal clinical language or casual patient speech.

#     Returns:
#         JSON string with keys:
#         - symptoms: list of medical symptoms found
#         - duration: how long each symptom has lasted
#         - severity: severity level of each symptom
#         - urgent: true if immediate attention may be required

#     Example:
#         Input:  "chest feels weird, tired walking to kitchen for 3 days"
#         Output: {
#                   "symptoms": ["chest discomfort", "fatigue"],
#                   "duration": ["3 days", "3 days"],
#                   "severity": ["unspecified", "mild"],
#                   "urgent": true
#                 }
#     """
#     if not clinical_note or not clinical_note.strip():
#         return json.dumps({"error": "No clinical note provided"})

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": clinical_note.strip()}
#             ],
#             temperature=0.1,   # low temperature for consistent structured output
#             max_tokens=500
#         )

#         raw = response.choices[0].message.content
#         raw = clean_json(raw)

#         # Validate it parses as JSON
#         result = json.loads(raw)

#         # Ensure all required fields exist
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

#         return json.dumps(result, indent=2)

#     except json.JSONDecodeError as e:
#         return json.dumps({
#             "error": "Failed to parse model output as JSON",
#             "details": str(e),
#             "raw_output": raw if 'raw' in locals() else "no output"
#         })
#     except Exception as e:
#         return json.dumps({
#             "error": "Extraction failed",
#             "details": str(e)
#         })


# @mcp.tool()
# def check_urgency(clinical_note: str) -> str:
#     """
#     Quickly check if a clinical note describes an urgent medical situation.

#     Returns a simple urgent/non-urgent assessment with reasoning.

#     Args:
#         clinical_note: Raw clinical text or patient description

#     Returns:
#         JSON string with:
#         - urgent: true/false
#         - reason: brief explanation of the urgency assessment
#         - recommended_action: suggested next step
#     """
#     if not clinical_note or not clinical_note.strip():
#         return json.dumps({"error": "No clinical note provided"})

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
#         return json.dumps(result, indent=2)

#     except Exception as e:
#         return json.dumps({
#             "error": "Urgency check failed",
#             "details": str(e)
#         })


# if __name__ == "__main__":
#     mcp.run(transport="stdio")

# ============================================================
# ClinicalDistill MCP Server
# ============================================================
# Two modes:
#   USE_FINETUNED = False → GPT-4o (fast, reliable, for demos)
#   USE_FINETUNED = True  → Fine-tuned Gemma on HuggingFace (research model)
#
# Toggle with: USE_FINETUNED = True/False
# ============================================================

from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from gradio_client import Client as GradioClient
import json
import re

# ============================================================
# TOGGLE THIS to switch between models
# False = GPT-4o (recommended for demos)
# True  = Your fine-tuned Gemma on HuggingFace Spaces
# ============================================================
USE_FINETUNED = False

# Initialize MCP server
# mcp = FastMCP("ClinicalDistill")
mcp = FastMCP(
    "ClinicalDistill",
    capabilities={
        "extensions": {
            "ai.promptopinion/fhir-context": {
                "scopes": [
                    {
                        "name": "patient/Patient.rs",
                        "required": True
                    },
                    {
                        "name": "patient/Condition.rs",
                        "required": False
                    },
                    {
                        "name": "patient/Observation.rs",
                        "required": False
                    }
                ]
            }
        }
    }
)

# Initialize clients
openai_client = OpenAI()  # uses OPENAI_API_KEY from environment
HF_SPACE_ID = "Janushi/ClinicalDistill"  # your HuggingFace Space

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


def normalize_result(result: dict) -> dict:
    """Ensure all required fields exist and arrays are same length"""
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

    return result


def extract_with_gpt(clinical_note: str) -> dict:
    """Use GPT-4o for extraction — fast and reliable"""
    response = openai_client.chat.completions.create(
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
    result["_model"] = "gpt-4o"
    return result


def extract_with_finetuned(clinical_note: str) -> dict:
    """Use fine-tuned Gemma via HuggingFace Space API"""
    gradio = GradioClient(HF_SPACE_ID)
    raw_output = gradio.predict(
        clinical_note,
        api_name="/extract"
    )

    # Space returns table HTML + JSON — extract the JSON part
    if isinstance(raw_output, (list, tuple)):
        # Second output is the raw JSON from the accordion
        raw = raw_output[1] if len(raw_output) > 1 else raw_output[0]
    else:
        raw = raw_output

    raw = clean_json(str(raw))
    result = json.loads(raw)
    result["_model"] = "ClinicalDistill-Gemma-1B (fine-tuned)"
    return result


# @mcp.tool()
# def extract_symptoms(clinical_note: str) -> str:
#     """
#     Extract structured clinical symptoms from unstructured medical text.

#     Converts casual or clinical patient descriptions into structured JSON
#     containing symptoms, duration, severity, and an urgency flag.

#     Args:
#         clinical_note: Raw clinical text or patient description.
#                       Can be formal clinical language or casual patient speech.

#     Returns:
#         JSON string with keys:
#         - symptoms: list of medical symptoms found
#         - duration: how long each symptom has lasted
#         - severity: severity level of each symptom
#         - urgent: true if immediate attention may be required
#         - _model: which model was used for extraction

#     Example:
#         Input:  "chest feels weird, tired walking to kitchen for 3 days"
#         Output: {
#                   "symptoms": ["chest discomfort", "fatigue"],
#                   "duration": ["3 days", "3 days"],
#                   "severity": ["unspecified", "mild"],
#                   "urgent": true,
#                   "_model": "gpt-4o"
#                 }
#     """
#     if not clinical_note or not clinical_note.strip():
#         return json.dumps({"error": "No clinical note provided"})

#     try:
#         if USE_FINETUNED:
#             result = extract_with_finetuned(clinical_note)
#         else:
#             result = extract_with_gpt(clinical_note)

#         result = normalize_result(result)
#         return json.dumps(result, indent=2)

#     except json.JSONDecodeError as e:
#         return json.dumps({
#             "error": "Failed to parse model output as JSON",
#             "details": str(e)
#         })
#     except Exception as e:
#         # Fallback to GPT if fine-tuned model fails
#         if USE_FINETUNED:
#             try:
#                 result = extract_with_gpt(clinical_note)
#                 result = normalize_result(result)
#                 result["_fallback"] = True
#                 result["_fallback_reason"] = str(e)
#                 return json.dumps(result, indent=2)
#             except Exception as fallback_error:
#                 return json.dumps({
#                     "error": "Both models failed",
#                     "details": str(fallback_error)
#                 })
#         return json.dumps({
#             "error": "Extraction failed",
#             "details": str(e)
#         })
@mcp.tool()
def extract_symptoms(
    clinical_note: str,
    fhir_server_url: str = None,      # X-FHIR-Server-URL header
    fhir_access_token: str = None,    # X-FHIR-Access-Token header  
    patient_id: str = None            # X-Patient-ID header
) -> str:
    """
    Extract structured clinical symptoms from unstructured medical text.
    
    Args:
        clinical_note: Raw clinical text or patient description
        fhir_server_url: FHIR server URL (injected by Prompt Opinion)
        fhir_access_token: FHIR access token (injected by Prompt Opinion)
        patient_id: Current patient ID (injected by Prompt Opinion)
    """
    if not clinical_note or not clinical_note.strip():
        return json.dumps({"error": "No clinical note provided"})

    try:
        if USE_FINETUNED:
            result = extract_with_finetuned(clinical_note)
        else:
            result = extract_with_gpt(clinical_note)

        result = normalize_result(result)
        
        # Add FHIR context to response if provided
        if patient_id:
            result["patient_id"] = patient_id
        if fhir_server_url:
            result["fhir_server"] = fhir_server_url
            
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def check_urgency(clinical_note: str) -> str:
    """
    Quickly check if a clinical note describes an urgent medical situation.

    Returns a simple urgent/non-urgent assessment with reasoning.
    Always uses GPT-4o for speed regardless of USE_FINETUNED setting.

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
        response = openai_client.chat.completions.create(
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


@mcp.tool()
def get_model_info() -> str:
    """
    Get information about which model is currently being used for extraction.

    Returns:
        JSON string with current model configuration and performance metrics
    """
    if USE_FINETUNED:
        info = {
            "active_model": "ClinicalDistill-Gemma-1B",
            "type": "Fine-tuned small LLM",
            "base_model": "google/gemma-3-1b-it",
            "fine_tuning": "LoRA (r=16, alpha=32)",
            "training_data": "145 GPT-4o generated clinical examples",
            "performance": {
                "valid_json_rate": "100%",
                "symptom_f1": 0.781,
                "urgent_accuracy": "85.7%"
            },
            "hosted_at": f"https://huggingface.co/spaces/{HF_SPACE_ID}",
            "model_card": "https://huggingface.co/Janushi/ClinicalDistill-Gemma-1B",
            "note": "Running on CPU — inference takes ~60 seconds"
        }
    else:
        info = {
            "active_model": "GPT-4o",
            "type": "Large language model via API",
            "provider": "OpenAI",
            "note": "Fast and reliable for production demos",
            "research_model": {
                "name": "ClinicalDistill-Gemma-1B",
                "description": "Fine-tuned 1B parameter model — set USE_FINETUNED=True to use",
                "performance": {
                    "valid_json_rate": "100%",
                    "symptom_f1": 0.781,
                    "urgent_accuracy": "85.7%"
                },
                "hosted_at": f"https://huggingface.co/spaces/{HF_SPACE_ID}"
            }
        }

    return json.dumps(info, indent=2)


# if __name__ == "__main__":
#     model_name = "Fine-tuned Gemma-1B" if USE_FINETUNED else "GPT-4o"
#     print(f"Starting ClinicalDistill MCP Server")
#     print(f"Active model: {model_name}")
#     print(f"Tools: extract_symptoms, check_urgency, get_model_info")
#     mcp.run(transport="stdio")
if __name__ == "__main__":
    import uvicorn
    model_name = "Fine-tuned Gemma-1B" if USE_FINETUNED else "GPT-4o"
    print(f"Starting ClinicalDistill MCP Server")
    print(f"Active model: {model_name}")
    print(f"Tools: extract_symptoms, check_urgency, get_model_info")
    
    app = mcp.sse_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)