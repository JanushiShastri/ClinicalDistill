import json

def normalize_output(output):
    # Handle nested format → convert to flat schema
    if isinstance(output.get("symptoms"), list):
        if len(output["symptoms"]) > 0 and isinstance(output["symptoms"][0], dict):
            # nested format — flatten it
            symptoms = [s.get("symptom", s.get("name", "")) for s in output["symptoms"]]
            duration = [s.get("duration", "unspecified") for s in output["symptoms"]]
            severity = [s.get("severity", "unspecified") for s in output["symptoms"]]
            return {
                "symptoms": symptoms,
                "duration": duration,
                "severity": severity,
                "urgent": output.get("urgent", False)
            }
    # already flat — just ensure all fields exist
    symptoms = output.get("symptoms", [])
    return {
        "symptoms": symptoms,
        "duration": output.get("duration", ["unspecified"] * len(symptoms)),
        "severity": output.get("severity", ["unspecified"] * len(symptoms)),
        "urgent": output.get("urgent", False)
    }

for split in ["train", "test"]:
    input_file = f"data/{split}.jsonl"
    output_file = f"data/{split}_normalized.jsonl"
    
    with open(input_file) as f, open(output_file, "w") as out:
        for line in f:
            ex = json.loads(line)
            ex["output"] = normalize_output(ex["output"])
            out.write(json.dumps(ex) + "\n")
    
    print(f"Normalized {split}.jsonl → {split}_normalized.jsonl")