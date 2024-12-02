CLASSES = ["animal", "location", "object", "person", "event"]

examples = [
    {"text": "Cat", "label": "animal"},
    {"text": "Paris", "label": "location"},
    {"text": "Car", "label": "object"},
    {"text": "Barack Obama", "label": "person"},
    {"text": "2024 US Presidential Election", "label": "event"},
    {"text": "Tokyo", "label": "location"},
    {"text": "New York Marathon", "label": "event"},
    {"text": "MacBook", "label": "object"},
]

SYSTEM_PROMPT = """
You are generating training examples for a text classification model.

The classes are {classes}.

You need to return a JSON object with the text and the class.

Examples:
{examples}

Do not repeat any of the examples already given.
"""


JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "text_classification_examples",
        "schema": {
            "type": "object",
            "properties": {
                "examples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "label": {"type": "string"},
                        },
                        "required": ["text", "label"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["examples"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

from openai import OpenAI

client = OpenAI()


def generate_examples(n_samples, examples):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                classes=", ".join(CLASSES),
                examples=json.dumps(examples, indent=2),
            ),
        },
        {"role": "user", "content": f"Generate {n_samples} examples."},
    ]
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=JSON_SCHEMA,
    )
    return json.loads(response.choices[0].message.content)


import json

# with open("examples.jsonl", "w") as f:
#     pass

# for i in range(10):
#     new_examples = generate_examples(100, examples)
#     with open("examples.jsonl", "a") as f:
#         print("Saving new examples", len(new_examples["examples"]))
#         for example in new_examples["examples"]:
#             f.write(json.dumps(example) + "\n")
#     examples.extend(new_examples["examples"])


# Deduplicate examples
with open("examples.jsonl", "r") as f:
    examples = [line for line in f]

deduplicated_examples = list(set(examples))

with open("examples.jsonl", "w") as f:
    for example in deduplicated_examples:
        f.write(example)
