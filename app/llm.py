import json

from pydantic import BaseModel

from .config import CHAT_MODEL, get_client


def _parse(raw: str, model_cls: type[BaseModel]) -> BaseModel:
    data = json.loads(raw)
    # Tolerate models that echo the schema and nest the values under "properties".
    if (isinstance(data, dict) and isinstance(data.get("properties"), dict)
            and not set(model_cls.model_fields) & set(data)):
        data = data["properties"]
    return model_cls.model_validate(data)


def json_call(system: str, user: str, model_cls: type[BaseModel]) -> BaseModel:
    """Chat call that must return JSON validating against model_cls (retries with feedback)."""
    client = get_client()
    schema = json.dumps(model_cls.model_json_schema())
    instructions = (
        f"{system}\n\n"
        f"The JSON Schema below only describes the required format. Do NOT return the schema. "
        f"Return one JSON object whose top-level keys are the fields themselves "
        f"(no \"properties\" wrapper, no \"type\", no \"$defs\").\n"
        f"Schema: {schema}"
    )
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user},
    ]
    last_err = None

    for _ in range(3):
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            response_format={"type": "json_object"},
            messages=messages,
        )
        raw = resp.choices[0].message.content or ""
        try:
            return _parse(raw, model_cls)
        except Exception as e:
            last_err = e
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": (
                f"That was invalid: {str(e)[:300]}. Return only the corrected JSON object "
                f"with the fields at the top level.")})
    raise RuntimeError(f"Model did not return valid {model_cls.__name__}: {last_err}")