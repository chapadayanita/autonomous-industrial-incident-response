"""
LLM incident-report step (explanation layer only) - uses Google Gemini (free tier).

The LLM NEVER makes decisions or triggers actions. Your existing ML model and
agents decide everything; this module only turns their evidence into a short
plain-language report for the human operator. If the API call fails for any
reason (no key, no internet, rate limit, retired model), a deterministic
template is used, so your demo never breaks.

Setup:
    pip install google-genai python-dotenv
    Put GEMINI_API_KEY=your-key in a .env file in the project root.
    (Get a free key at https://aistudio.google.com/app/apikey)
Never hard-code the key or commit it to GitHub.

Optional: set GEMINI_MODEL in .env to force one model (e.g. GEMINI_MODEL=gemini-3.5-flash-lite).
"""
import os

try:  # load GEMINI_API_KEY from the .env file in the project root
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

# Tried in order until one works (model names change often; older ones get retired).
_FORCED = os.getenv("GEMINI_MODEL")
MODELS = [_FORCED] if _FORCED else [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
]

SYSTEM_PROMPT = (
    "You write short incident reports for industrial plant operators. "
    "Use ONLY the facts given. Do not invent readings, causes, or actions. "
    "Write at most 100 words: what happened, the evidence, the recommended "
    "procedure, and whether human escalation is needed."
)


def template_report(evidence: dict, procedure: str, escalated: bool) -> str:
    """Deterministic fallback (also a useful baseline to compare against)."""
    return (
        f"Incident: anomaly detected on {evidence.get('sensor', 'unknown sensor')}. "
        f"Evidence: {evidence.get('summary', 'n/a')}. "
        f"Recommended procedure: {procedure}. "
        f"Human escalation: {'REQUIRED' if escalated else 'not required'}."
    )


def generate_incident_report(evidence: dict, procedure: str, escalated: bool) -> tuple[str, str]:
    """Returns (report_text, source) where source is 'llm' or 'template'."""
    user_msg = (
        f"Evidence: {evidence}\n"
        f"Retrieved maintenance procedure: {procedure}\n"
        f"Human escalation required: {'yes' if escalated else 'no'}""
    )
    try:
        from google import genai  # imported here so the fallback works without the package
        from google.genai import types

        client = genai.Client()  # reads GEMINI_API_KEY from the environment
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            max_output_tokens=1024,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
    except Exception as e:
        print(f"[report] Gemini setup failed, using template ({type(e).__name__}: {str(e)[:120]})")
        return template_report(evidence, procedure, escalated), "template"

    for model in MODELS:
        try:
            resp = client.models.generate_content(model=model, contents=user_msg, config=config)
            text = (resp.text or "").strip()
            if text:
                return text, "llm"
        except Exception as e:  # retired model, quota, no network, bad key...
            print(f"[report] {model} failed ({type(e).__name__}: {str(e)[:100]})")
    print("[report] All models failed, using template")
    return template_report(evidence, procedure, escalated), "template"


if __name__ == "__main__":
    demo_evidence = {
        "sensor": "pump_3_temperature",
        "summary": "temperature rose from 62C to 91C in 4 minutes; vibration above threshold",
    }
    report, source = generate_incident_report(
        demo_evidence, "Pump overheating: shut down feed valve, inspect bearings", True
    )
    print(f"[{source}] {report}")