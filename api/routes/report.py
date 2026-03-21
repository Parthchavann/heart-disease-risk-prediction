"""
Medical report extraction endpoint.
Accepts blood test PDFs / lab images and uses Gemini Vision to extract
clinical values that map directly to PatientDataRequest fields.
"""

import asyncio
import base64
import json
import logging
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/report", tags=["report"])

MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

ALLOWED_MIME_TYPES = {
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/png": "image/png",
    "application/pdf": "application/pdf",
}

EXTRACTION_PROMPT = """You are a medical data extraction assistant. Analyze this medical report (blood test, ECG, lab report, or cardiac imaging) and extract the following clinical values if present.

Return ONLY a valid JSON object with this exact structure (use null for values not found):
{
  "extracted": {
    "age": null,
    "sex": null,
    "cp": null,
    "trestbps": null,
    "chol": null,
    "fbs": null,
    "restecg": null,
    "thalach": null,
    "exang": null,
    "oldpeak": null,
    "slope": null,
    "ca": null,
    "thal": null
  },
  "confidence": {
    "age": null,
    "sex": null,
    "cp": null,
    "trestbps": null,
    "chol": null,
    "fbs": null,
    "restecg": null,
    "thalach": null,
    "exang": null,
    "oldpeak": null,
    "slope": null,
    "ca": null,
    "thal": null
  },
  "notes": ""
}

Field definitions and encoding:
- age: patient age in years (integer 18-120)
- sex: 0=female, 1=male (integer)
- cp: chest pain type — 0=typical angina, 1=atypical angina, 2=non-anginal pain, 3=asymptomatic (integer 0-3)
- trestbps: resting blood pressure in mm Hg (integer 80-200). Use systolic value.
- chol: serum cholesterol in mg/dl (integer 100-600). Use total cholesterol.
- fbs: fasting blood sugar > 120 mg/dl — 0=false, 1=true (integer)
- restecg: resting ECG results — 0=normal, 1=ST-T wave abnormality, 2=left ventricular hypertrophy (integer 0-2)
- thalach: maximum heart rate achieved in bpm (integer 60-220)
- exang: exercise-induced angina — 0=no, 1=yes (integer)
- oldpeak: ST depression induced by exercise relative to rest (float 0-10)
- slope: slope of peak exercise ST segment — 0=upsloping, 1=flat, 2=downsloping (integer 0-2)
- ca: number of major vessels colored by fluoroscopy (integer 0-4)
- thal: thalassemia — 0=normal, 1=fixed defect, 2=reversible defect, 3=not described (integer 0-3)

Confidence values should be: "high" (clearly stated in report), "medium" (inferred/calculated), "low" (estimated from context), or null (not found).

For notes: briefly mention what type of report this is and any important caveats.

Return ONLY the JSON. No markdown, no explanation."""


class ExtractedReport(BaseModel):
    extracted: dict
    confidence: dict
    notes: str
    fields_found: int
    file_type: str


def _call_gemini_vision(file_bytes: bytes, mime_type: str) -> dict:
    """Call Gemini Vision API to extract medical values from file."""
    try:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        part = genai_types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[part, EXTRACTION_PROMPT],
        )

        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        logger.error(f"Gemini returned non-JSON: {e}")
        raise ValueError("Could not parse extraction response from AI model.")
    except Exception as e:
        logger.error(f"Gemini vision call failed: {e}")
        raise


@router.post("/extract", response_model=ExtractedReport)
async def extract_medical_report(file: UploadFile = File(...)):
    """
    Extract clinical values from a medical report image or PDF.

    Accepts: JPEG, PNG, PDF (max 10 MB)
    Returns: Extracted field values + confidence + notes
    Files are processed in memory — never saved to disk.
    """
    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Medical report extraction requires GEMINI_API_KEY to be configured."
        )

    # Validate file type
    content_type = file.content_type or ""
    # Normalise — browsers sometimes send "image/jpg"
    mime_type = ALLOWED_MIME_TYPES.get(content_type)
    if not mime_type:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Accepted: JPEG, PNG, PDF."
        )

    # Read file into memory
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes) // 1024} KB). Maximum is 10 MB."
        )
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info(f"Processing medical report: {file.filename} ({len(file_bytes)} bytes, {mime_type})")

    try:
        result = await asyncio.to_thread(_call_gemini_vision, file_bytes, mime_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Report extraction failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract data from report. Please try again.")

    extracted = result.get("extracted", {})
    confidence = result.get("confidence", {})
    notes = result.get("notes", "")

    # Count non-null extracted fields
    fields_found = sum(1 for v in extracted.values() if v is not None)

    # Clamp values to valid ranges to prevent downstream validation errors
    ranges = {
        "age": (18, 120), "sex": (0, 1), "cp": (0, 3),
        "trestbps": (80, 200), "chol": (100, 600), "fbs": (0, 1),
        "restecg": (0, 2), "thalach": (60, 220), "exang": (0, 1),
        "oldpeak": (0.0, 10.0), "slope": (0, 2), "ca": (0, 4), "thal": (0, 3),
    }
    for field, (lo, hi) in ranges.items():
        val = extracted.get(field)
        if val is not None:
            try:
                val = float(val)
                extracted[field] = max(lo, min(hi, val))
            except (TypeError, ValueError):
                extracted[field] = None
                confidence[field] = None

    logger.info(f"Extraction complete: {fields_found} fields found from {file.filename}")

    return ExtractedReport(
        extracted=extracted,
        confidence=confidence,
        notes=notes,
        fields_found=fields_found,
        file_type=mime_type,
    )
