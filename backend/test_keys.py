import os
import sys

# Ensure backend path is configured for importing config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from google import genai
from config import GEMINI_API_KEYS, MODEL_FLASH

def test_api_keys():
    print(f"==========================================")
    print(f"🔑 Gemini API Keys Test")
    print(f"Total keys configured: {len(GEMINI_API_KEYS)}")
    print(f"Testing model: {MODEL_FLASH}")
    print(f"==========================================\n")

    for i, api_key in enumerate(GEMINI_API_KEYS):
        masked_key = f"{api_key[:5]}...{api_key[-5:]}" if len(api_key) > 10 else "Invalid format"
        print(f"Testing Key {i+1}/{len(GEMINI_API_KEYS)}: [{masked_key}]")
        
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=MODEL_FLASH,
                contents="Reply with the single word: OK",
            )
            
            result = response.text.strip()
            if result.upper() == "OK":
                print(f"  ✅ SUCCESS: Key is active and working.\n")
            else:
                print(f"  ⚠️ UNEXPECTED RESPONSE: {result}\n")
                
        except Exception as e:
            err_msg = str(e)
            if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                print(f"  ❌ RATE LIMITED: 429 RESOURCE_EXHAUSTED. Quota exceeded or too many requests.\n")
            elif "API_KEY_INVALID" in err_msg or "400" in err_msg:
                print(f"  ❌ INVALID KEY: Authentication failed.\n")
            else:
                print(f"  ❌ ERROR: {err_msg}\n")

if __name__ == "__main__":
    if not GEMINI_API_KEYS:
        print("❌ No API keys found in config.py. Please configure them first.")
    else:
        test_api_keys()
