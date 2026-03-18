"""GLiNER2 vLLM Processor — handles preprocessing and postprocessing.

Ports the SchemaTransformer logic from gliner2.processor for vLLM use.
Preprocessing: Schema + text → tokenized input with mapped_indices.
Postprocessing: Raw output tensor → structured extraction results.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Dict, List

# ==================================================================
# Text Splitter (mirrors gliner2.processor.WhitespaceTokenSplitter)
# ==================================================================

WORD_PATTERN = re.compile(
    r"""(?:https?://[^\s]+|www\.[^\s]+)
    |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
    |@[a-z0-9_]+
    |\w+(?:[-_]\w+)*
    |\S""",
    re.VERBOSE | re.IGNORECASE,
)


def split_words(text: str, lower: bool = True):
    t = text.lower() if lower else text
    return [(m.group(), m.start(), m.end()) for m in WORD_PATTERN.finditer(t)]


# ==================================================================
# Special Tokens
# ==================================================================

SEP_STRUCT = "[SEP_STRUCT]"
SEP_TEXT = "[SEP_TEXT]"
P_TOKEN = "[P]"
C_TOKEN = "[C]"
E_TOKEN = "[E]"
R_TOKEN = "[R]"
L_TOKEN = "[L]"
EXAMPLE_TOKEN = "[EXAMPLE]"
OUTPUT_TOKEN = "[OUTPUT]"
DESC_TOKEN = "[DESCRIPTION]"

SPECIAL_TOKENS = [
    SEP_STRUCT, SEP_TEXT, P_TOKEN, C_TOKEN, E_TOKEN,
    R_TOKEN, L_TOKEN, EXAMPLE_TOKEN, OUTPUT_TOKEN, DESC_TOKEN,
]


# ==================================================================
# Schema Processing (inference-only, no training augmentation)
# ==================================================================

def build_schema_for_entities(entity_types: List[str]) -> Dict:
    """Build schema dict for entity extraction."""
    schema = {
        "json_structures": [],
        "classifications": [],
        "entities": OrderedDict({e: "" for e in entity_types}),
        "relations": [],
    }
    return schema


def build_schema_for_relations(relation_types: List[str]) -> Dict:
    """Build schema dict for relation extraction."""
    schema = {
        "json_structures": [],
        "classifications": [],
        "entities": OrderedDict(),
        "relations": [{r: {"head": "", "tail": ""}} for r in relation_types],
    }
    return schema


def build_schema_for_json(structures: Dict[str, List[str]]) -> Dict:
    """Build schema dict for JSON structure extraction."""
    json_structures = []
    for parent, fields in structures.items():
        json_structures.append({parent: {f: "" for f in fields}})
    schema = {
        "json_structures": json_structures,
        "classifications": [],
        "entities": OrderedDict(),
        "relations": [],
    }
    return schema


def build_schema_for_classification(tasks: Dict) -> Dict:
    """Build schema dict for classification tasks."""
    classifications = []
    for task_name, labels in tasks.items():
        if isinstance(labels, dict) and "labels" in labels:
            config = labels.copy()
            config["task"] = task_name
            config.setdefault("true_label", ["N/A"])
        else:
            config = {
                "task": task_name,
                "labels": labels,
                "true_label": ["N/A"],
            }
        classifications.append(config)
    schema = {
        "json_structures": [],
        "classifications": classifications,
        "entities": OrderedDict(),
        "relations": [],
    }
    return schema


# ==================================================================
# Schema → Token Sequence
# ==================================================================

def _transform_schema(parent, fields, child_prefix, prompt=None,
                      label_descriptions=None, example_mode="none"):
    """Transform a single schema into a token sequence."""
    prompt_str = parent
    if prompt:
        prompt_str = f"{parent}: {prompt}"

    if example_mode in ("descriptions", "both") and label_descriptions:
        for label, desc in label_descriptions.items():
            if label in fields:
                prompt_str += f" {DESC_TOKEN} {label}: {desc}"

    tokens = ["(", P_TOKEN, prompt_str, "("]
    for field in fields:
        tokens.extend([child_prefix, field])
    tokens.extend([")", ")"])
    return tokens


def infer_schemas_from_dict(schema: Dict) -> Dict:
    """Infer schema token sequences and task types from schema dict (inference mode)."""
    schemas = []
    labels = []
    types = []

    # Process JSON structures
    if "json_structures" in schema:
        for item in schema["json_structures"]:
            for parent, fields in item.items():
                field_names = list(fields.keys())
                schemas.append(_transform_schema(parent, field_names, C_TOKEN))
                count = sum(1 for v in fields.values() if v != "")
                labels.append([max(1, count), []])
                types.append("json_structures")

    # Process entities
    if "entities" in schema and schema["entities"]:
        entity_fields = list(schema["entities"].keys())
        schemas.append(_transform_schema("entities", entity_fields, E_TOKEN))
        labels.append([1, []])
        types.append("entities")

    # Process relations
    if "relations" in schema:
        for item in schema["relations"]:
            for parent, fields in item.items():
                field_names = list(fields.keys())
                schemas.append(_transform_schema(parent, field_names, R_TOKEN))
                labels.append([1, []])
                types.append("relations")

    # Process classifications
    if "classifications" in schema:
        for cls_item in schema["classifications"]:
            task = cls_item["task"]
            cls_labels = cls_item["labels"]
            descs = cls_item.get("label_descriptions", {})
            mode = "descriptions" if descs else "none"
            schemas.append(_transform_schema(
                task, cls_labels, L_TOKEN,
                prompt=cls_item.get("prompt"),
                label_descriptions=descs,
                example_mode=mode,
            ))
            labels.append([])
            types.append("classifications")

    return {"schemas": schemas, "task_types": types, "structure_labels": labels}


# ==================================================================
# Input Formatting
# ==================================================================

def format_input_with_mapping(tokenizer, schema_tokens_list, text_tokens):
    """Format schema + text into token IDs with segment mappings."""
    combined = []
    for struct in schema_tokens_list:
        combined.extend(struct)
        combined.append(SEP_STRUCT)
    if combined:
        combined.pop()
    combined.append(SEP_TEXT)
    combined.extend(text_tokens)

    subwords = []
    mappings = []
    num_schemas = len(schema_tokens_list)
    text_schema_idx = num_schemas
    current_schema = 0
    found_sep = False

    for orig_idx, token in enumerate(combined):
        if token == SEP_TEXT:
            seg_type = "sep"
            schema_idx = text_schema_idx
            found_sep = True
        elif not found_sep:
            seg_type = "schema"
            schema_idx = current_schema
            if token == SEP_STRUCT:
                current_schema += 1
        else:
            seg_type = "text"
            schema_idx = text_schema_idx

        sub_tokens = tokenizer.tokenize(token)
        subwords.extend(sub_tokens)
        mappings.extend([(seg_type, orig_idx, schema_idx)] * len(sub_tokens))

    input_ids = tokenizer.convert_tokens_to_ids(subwords)
    return {
        "input_ids": input_ids,
        "mapped_indices": mappings,
    }


# ==================================================================
# Main Preprocessing
# ==================================================================

def preprocess(tokenizer, text: str, schema: Dict, token_pooling: str = "first"):
    """Preprocess text + schema for vLLM inference.

    Returns a dict with all data needed for vLLM embed call.
    """
    # Ensure text ends with punctuation
    if text and not text.endswith((".", "!", "?")):
        text = text + "."
    elif not text:
        text = "."

    # Split text
    word_triples = split_words(text, lower=True)
    text_tokens = [w[0] for w in word_triples]
    start_mapping = [w[1] for w in word_triples]
    end_mapping = [w[2] for w in word_triples]

    # Infer schemas
    processed = infer_schemas_from_dict(schema)
    schema_tokens_list = processed["schemas"]
    task_types = processed["task_types"]

    # Format input
    fmt = format_input_with_mapping(tokenizer, schema_tokens_list, text_tokens)

    # Get special token IDs
    special_ids = {}
    for tok in SPECIAL_TOKENS:
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tok))
        if ids:
            special_ids[tok] = ids[0]

    return {
        "input_ids": fmt["input_ids"],
        "mapped_indices": fmt["mapped_indices"],
        "schema_tokens_list": schema_tokens_list,
        "task_types": task_types,
        "text_tokens": text_tokens,
        "schema_count": len(schema_tokens_list),
        "original_text": text,
        "start_mapping": start_mapping,
        "end_mapping": end_mapping,
        "special_token_ids": special_ids,
        "token_pooling": token_pooling,
        "schema_dict": schema,
    }


# ==================================================================
# Postprocessing
# ==================================================================

def decode_output(raw_output, schema: Dict, task_types: List[str] = None) -> Dict:
    """Decode raw vLLM output tensor back into structured results.

    The output is a JSON-encoded byte tensor prepended by length.
    """

    if isinstance(raw_output, list):
        data = raw_output
    elif hasattr(raw_output, 'tolist'):
        data = raw_output.tolist()
    else:
        data = list(raw_output)

    length = int(data[0])
    byte_data = bytes([int(b) for b in data[1:length + 1]])
    results = json.loads(byte_data.decode("utf-8"))
    return results


def format_results(results: Dict, include_confidence: bool = False) -> Dict:
    """Format raw results into user-friendly output."""
    formatted = {}
    relations = {}

    for key, value in results.items():
        if not isinstance(value, dict) or "type" not in value:
            formatted[key] = value
            continue

        result_type = value["type"]

        if result_type == "classification":
            logits = value.get("logits", [])
            labels = value.get("labels", [])
            import torch
            probs = torch.softmax(torch.tensor(logits), dim=-1)
            best = int(probs.argmax().item())
            if include_confidence:
                formatted[key] = {"label": labels[best], "confidence": probs[best].item()}
            else:
                formatted[key] = labels[best]

        elif result_type == "entities":
            entities = value.get("entities", {})
            ent_formatted = {}
            for name, spans in entities.items():
                if include_confidence:
                    ent_formatted[name] = spans
                else:
                    ent_formatted[name] = [s["text"] for s in spans if isinstance(s, dict)]
            formatted["entities"] = ent_formatted

        elif result_type == "relations":
            instances = value.get("instances", [])
            rel_list = []
            for inst in instances:
                if include_confidence:
                    rel_list.append(inst)
                else:
                    rel_list.append(tuple(v["text"] for v in inst.values() if v))
            relations[key] = rel_list

        elif result_type == "json_structures":
            instances = value.get("instances", [])
            struct_list = []
            for inst in instances:
                if include_confidence:
                    struct_list.append(inst)
                else:
                    struct_list.append({
                        k: v["text"] if isinstance(v, dict) and "text" in v else v
                        for k, v in inst.items()
                    })
            formatted[key] = struct_list

    if relations:
        formatted["relation_extraction"] = relations

    return formatted
