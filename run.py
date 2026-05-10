"""Run the API with PYTHONPATH set correctly. Just do: python run.py"""
import os
import sys

os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
