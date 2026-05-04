from mcp.server.fastmcp import FastMCP
from tools.patient_age_tool import get_patient_age
from tools.patient_allergies_tool import get_patient_allergies
from tools.patient_id_tool import find_patient_id

# ── ADDED: ClinicalDistill custom tools ───────────────────────────────────
from tools.clinical_distill_tool import (
    extract_clinical_symptoms,   # Extract structured symptoms from text
    check_clinical_urgency,      # Quick triage urgency assessment
    check_patient_documents,     # List patient FHIR documents
    analyze_patient_notes        # Full pipeline: read → extract → save
)
# ─────────────────────────────────────────────────────────────────────────

# MODIFIED: renamed from "Python Template" to "ClinicalDistill"
mcp = FastMCP("ClinicalDistill", stateless_http=True, host="0.0.0.0")

_original_get_capabilities = mcp._mcp_server.get_capabilities

# ── MODIFIED: patched to declare Prompt Opinion FHIR extension ────────────
# Added DocumentReference.rs and Condition.crus scopes vs original
# which only had Patient, Observation, MedicationStatement, Condition.rs
def _patched_get_capabilities(notification_options, experimental_capabilities):
    caps = _original_get_capabilities(notification_options, experimental_capabilities)
    caps.model_extra["extensions"] = {
        "ai.promptopinion/fhir-context": {
            "scopes": [
                {"name": "patient/Patient.rs", "required": True},
                {"name": "patient/Observation.rs"},
                {"name": "patient/MedicationStatement.rs"},
                {"name": "patient/Condition.crus"},           # MODIFIED: crus for write access
                {"name": "patient/DocumentReference.rs",      # ADDED: read patient documents
                 "required": True},
            ]
        }
    }
    return caps
# ─────────────────────────────────────────────────────────────────────────

mcp._mcp_server.get_capabilities = _patched_get_capabilities

# ── ADDED: ClinicalDistill tools (our custom tools) ───────────────────────
mcp.tool(
    name="ExtractClinicalSymptoms",
    description="Extracts structured symptoms, duration, severity and urgency from unstructured clinical notes or patient descriptions."
)(extract_clinical_symptoms)

mcp.tool(
    name="CheckClinicalUrgency",
    description="Quickly assesses whether a clinical note describes an urgent medical situation requiring immediate attention."
)(check_clinical_urgency)

mcp.tool(
    name="CheckPatientDocuments",
    description="Checks what clinical documents exist for a patient in their FHIR record."
)(check_patient_documents)

mcp.tool(
    name="AnalyzePatientNotes",
    description="Fully automated: reads patient FHIR documents, extracts symptoms, and saves them back as FHIR Conditions. Provide patient ID."
)(analyze_patient_notes)
# ─────────────────────────────────────────────────────────────────────────

# Built-in tools from po-community-mcp starter repo (unchanged)
mcp.tool(name="GetPatientAge", description="Gets the age of a patient.")(get_patient_age)
mcp.tool(name="GetPatientAllergies", description="Gets the known allergies of a patient.")(get_patient_allergies)
mcp.tool(name="FindPatientId", description="Finds a patient id given a first name and last name")(find_patient_id)