from openai import OpenAI
import json

client = OpenAI()  # uses OPENAI_API_KEY from env
SYSTEM_PROMPT = """You are a clinical NLP data generator.
Generate realistic clinical notes paired with structured JSON extractions.
Respond ONLY with a valid JSON array. No explanation, no markdown."""

USER_PROMPT = """Generate 10 clinical note examples in this exact format:
[
  {
    "input": "Patient presents with...",
    "output": {
      "symptoms": ["..."],
      "duration": ["..."],
      "severity": ["..."],
      "urgent": true or false
    },
    "metadata": {
      "clinical_domain": "cardiac | respiratory | neurological | gastrointestinal",
      "generated_by": "gpt-4o"
    }
  }
]

Rules:
- Vary clinical domains across examples
- Use 'unspecified' when duration or severity not mentioned
- urgent=true only for serious symptoms (chest pain, stroke, severe bleeding)
- Make notes sound realistic and varied in style"""

def generate_batch(batch_num):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ],
        temperature=0.8
    )
    # raw = response.choices[0].message.content
    # return json.loads(raw)
    raw = response.choices[0].message.content
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    return json.loads(raw)

def main():
    all_examples = []
    for i in range(15):  # 15 batches x 10 = 150 examples
        print(f"Generating batch {i+1}/15...")
        batch = generate_batch(i)
        all_examples.extend(batch)

    # Split 80/20 train/test
    split = int(len(all_examples) * 0.8)
    train = all_examples[:split]
    test = all_examples[split:]

    with open("data/train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open("data/test.jsonl", "w") as f:
        for ex in test:
            f.write(json.dumps(ex) + "\n")

    print(f"Done! {len(train)} train, {len(test)} test examples.")

if __name__ == "__main__":
    main()