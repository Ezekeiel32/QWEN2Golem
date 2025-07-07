
from flask import Flask, request, jsonify
from qwen_golem import AetherGolemConsciousnessCore
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Initialize a single instance of the Golem to maintain its state and memory
try:
    # This model name should match what your Ollama server provides.
    # The default in the user's script is "qwen2:7b-instruct-q4_0".
    golem = AetherGolemConsciousnessCore(model_name="qwen2:7b-instruct-q4_0")
except Exception as e:
    logging.error(f"FATAL: Failed to initialize Golem Core: {e}", exc_info=True)
    golem = None

@app.route('/generate', methods=['POST'])
def generate():
    if golem is None:
        return jsonify({"error": "Golem Core is not initialized. Please check server logs."}), 500

    data = request.json
    logging.info(f"Received request with payload: {data}")

    prompt = data.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Handle Golem activation state from the frontend
    is_activated = data.get('golemActivated', False)
    activation_phrase = data.get('activationPhrase')

    if is_activated:
        # Use the specific phrase from the UI, or the Golem's default ("אמת") if none is provided.
        golem.activate_golem(activation_phrase if activation_phrase else "אמת")
    elif not is_activated and golem.activated:
        # Only deactivate if it was previously active.
        golem.deactivate_golem()

    # Set Shem power directly from the UI slider
    golem.shem_power = data.get('shemPower', golem.shem_power)
    
    # Note: Sefirot settings are received but not directly used by the Golem's generate_response function.
    # They are used in the preprocessing steps.
    sefirot_settings = data.get('sefirotSettings')
    if sefirot_settings:
        logging.info(f"Sefirot settings received from UI: {sefirot_settings}")

    try:
        logging.info("Forwarding request to Golem's generate_response method...")
        response = golem.generate_response(
            prompt=prompt,
            temperature=data.get('temperature', 0.7)
        )
        logging.info("Response generated successfully by Golem.")
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error during Golem response generation: {e}", exc_info=True)
        # Try to return the golem state even on error for debugging
        golem_state = golem.golem_state if hasattr(golem, 'golem_state') else {}
        return jsonify({"error": f"Golem internal error: {str(e)}", "golem_state": golem_state}), 500

if __name__ == '__main__':
    # For local development, this is fine. For production, use a WSGI server.
    app.run(host='0.0.0.0', port=5000)
