import json
import os
from typing import Any, Dict

import boto3


def _use_bedrock() -> bool:
    """
    Returns True if Bedrock is configured (either via API key or model ID env vars).
    """
    return bool(os.environ.get("BEDROCK_MODEL_ID") or os.environ.get("AWS_BEDROCK_API_KEY"))


def _call_bedrock_model(prompt_obj: Dict[str, Any]) -> str:
    """
    Calls Amazon Bedrock with the correct input schema depending on the selected model.
    Supports Titan and Anthropic (Claude) models.
    """
    import json

    model_id = os.environ.get("BEDROCK_MODEL_ID", "amazon.titan-text-express-v1")
    region = os.environ.get("AWS_REGION", "us-east-1")

    # Detect authentication type: API key or default credentials
    if os.getenv("AWS_BEDROCK_API_KEY"):
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=None,  # API key handled internally by Bedrock
        )
    else:
        bedrock = boto3.client("bedrock-runtime", region_name=region)

    # Titan model => simple schema
    if model_id.startswith("amazon.titan"):
        prompt_text = (
            f"System: {prompt_obj.get('system','')}\n\n"
            f"Question: {prompt_obj.get('question','')}\n\n"
            f"Context: {prompt_obj.get('context','')}\n"
        )
        body = json.dumps(
            {
                "inputText": prompt_text,
                "textGenerationConfig": {
                    "maxTokenCount": 512,
                    "temperature": 0.3,
                },
            }
        )

    # Claude / Anthropic => chat schema
    else:
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 800,
                "temperature": 0.3,
                "system": prompt_obj.get("system", ""),
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": json.dumps(prompt_obj)}],
                    }
                ],
            }
        )

    # Send request
    try:
        resp = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        payload = json.loads(resp["body"].read())

        # Titan returns: {'results':[{'outputText':'...'}]}
        if "results" in payload:
            return payload["results"][0].get("outputText", "")

        # Claude returns: {'content':[{'text':'...'}]}
        if "content" in payload:
            return payload["content"][0].get("text", "")

        return "(no response text found)"

    except Exception as e:
        # If anything fails, gracefully fall back
        return f"(LLM fallback) {str(e)}\n" + _fallback_local(prompt_obj)


def _fallback_local(prompt_obj: Dict[str, Any]) -> str:
    """
    Simple offline fallback mode for demos and local runs.
    """
    q = prompt_obj.get("question", "")
    mode = prompt_obj.get("mode", "Patient")
    ctx = prompt_obj.get("context", "")
    data_hint = "\n\n(Data available)" if prompt_obj.get("data_preview") else ""
    style = (
        "(clinician mode: concise, factual)"
        if mode == "Clinician"
        else "(patient mode: simple, reassuring)"
    )

    suggestion = ""
    if any(tok in q.lower() for tok in ["plot", "chart", "trend", "visual"]):
        suggestion = "\n\nSuggestion: Go to the Visualize tab and select a metric to see the trend."

    ctx_hint = f"\n\nContext used:\n{ctx[:600]}" if ctx else ""
    return f"Answer {style}:\n- Your question: {q}{data_hint}{ctx_hint}{suggestion}"


def llm_answer(prompt_obj: Dict[str, Any]) -> str:
    """
    Main entry point used by the Streamlit chat module.
    Automatically uses Bedrock if configured, otherwise fallback.
    """
    try:
        if _use_bedrock():
            return _call_bedrock_model(prompt_obj)
        return _fallback_local(prompt_obj)
    except Exception as e:
        return f"(LLM fallback) {str(e)}\n" + _fallback_local(prompt_obj)
