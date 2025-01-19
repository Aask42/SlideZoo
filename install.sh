#!/usr/bin/env bash

# 1) Create a virtual environment called "venv"
python3 -m venv venv

# 2) Activate the environment
#    On Windows, you'd do "venv\Scripts\activate.bat"
#    On Unix/macOS, it's:
source venv/bin/activate

# 3) Install packages from requirements.txt
pip install --upgrade pip
pip install -r requirements.txt

echo "=================================================="
echo "Virtual environment setup complete."
echo "Remember to 'source venv/bin/activate' (or use the equivalent on Windows)."
echo ""
echo "If you want AI image generation, set your API key via:"
echo "  export OPENAI_API_KEY=\"sk-XXXX\""
echo "Then run:"
echo "  python slideshow.py"
echo "=================================================="
