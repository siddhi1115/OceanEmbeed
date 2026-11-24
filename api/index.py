"""
api/index.py  — Vercel Python serverless entry point
Imports the FastAPI app from backend/main_vercel.py
Vercel auto-discovers this file from the api/ directory.
"""
import sys
import os

# Make backend/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.main_vercel import app  # noqa: F401 — Vercel picks up `app`
