"""
Check available Google Generative AI models.
"""

import os
import sys

try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: google-generativeai not installed. Run: pip install google-generativeai")
    sys.exit(1)

# Get API key from environment variable
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("ERROR: GOOGLE_API_KEY environment variable not set.")
    print("Set it with: $env:GOOGLE_API_KEY = 'your-key-here'")
    sys.exit(1)

print(f"Using API key: {api_key[:10]}...***")
genai.configure(api_key=api_key)

print("\nDostępne modele generatywne:")
print("-" * 60)

try:
    models = genai.list_models()
    count = 0
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            print(f"  • {m.name}")
            count += 1
    print("-" * 60)
    print(f"Razem: {count} modeli")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    sys.exit(1)
