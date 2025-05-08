from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoTokenizer
import logging
import torch  # Added for GPU support check

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize tokenizer and model for paraphrasing
MODEL_NAME = "humarin/chatgpt_paraphraser_on_T5_base"
paraphraser = None
tokenizer = None
MODEL_MAX_INPUT_LENGTH = 512  # Default for many T5 models, adjust if known
MAX_CHAR_LENGTH = 2000  # Added as initial quick check before tokenization

try:
    logger.info(f"Attempting to load tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.info("Tokenizer loaded successfully")
    
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
    paraphraser = pipeline(
        "text2text-generation",
        model=MODEL_NAME,
        tokenizer=tokenizer,
        device=device
    )
    logger.info(f"Model loaded on {'GPU' if device == 0 else 'CPU'}")
    
    if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length:
        MODEL_MAX_INPUT_LENGTH = tokenizer.model_max_length
    logger.info(f"Paraphrasing model loaded successfully. Max input tokens: {MODEL_MAX_INPUT_LENGTH}")
except Exception as e:
    logger.error(f"Failed to load model or tokenizer: {str(e)}", exc_info=True)
    paraphraser = None
    tokenizer = None

def safe_paraphrase(text, num_return_sequences=1):
    """Wrapper with error handling and input length check for paraphrasing"""
    if not paraphraser or not tokenizer:
        logger.warning("Paraphraser or tokenizer not available.")
        return {"error": "Paraphrasing model is not available. Please try again later."}, 503

    try:
        # Initial quick check by character length
        if len(text) > MAX_CHAR_LENGTH:
            logger.warning(f"Input text ({len(text)} chars) exceeds maximum character length ({MAX_CHAR_LENGTH} chars)")
            return {
                "error": f"Input text is too long ({len(text)} characters). Please reduce it to {MAX_CHAR_LENGTH} characters or less."
            }, 413

        # Tokenize the input to check its length
        inputs = tokenizer(text, return_tensors="pt", truncation=False)
        input_token_count = inputs.input_ids.shape[1]
        logger.info(f"Input text token count: {input_token_count}")

        if input_token_count > MODEL_MAX_INPUT_LENGTH:
            logger.warning(f"Input text ({input_token_count} tokens) exceeds model's maximum input length ({MODEL_MAX_INPUT_LENGTH} tokens)")
            return {
                "error": f"Input text is too long ({input_token_count} tokens). Please reduce it to {MODEL_MAX_INPUT_LENGTH} tokens or less."
            }, 413

        # Generate paraphrases
        result = paraphraser(
            text,
            max_length=500,  # Max length for the generated paraphrase
            num_beams=5,
            num_return_sequences=num_return_sequences,
            temperature=0.7,
            repetition_penalty=2.5,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            early_stopping=True
        )
        
        paraphrases = [res['generated_text'] for res in result]
        return {"paraphrases": paraphrases}, 200
        
    except Exception as e:
        logger.error(f"Paraphrasing failed: {str(e)}", exc_info=True)
        return {"error": "An unexpected error occurred during paraphrasing."}, 500

@app.route("/")
def home():
    return render_template("paraphrase.html")

@app.route("/test-model")
def test_model():
    """Endpoint to verify model is loaded and working"""
    if not paraphraser:
        return jsonify({
            "status": "Model not loaded",
            "error": "Check server logs for model loading issues"
        }), 503
    
    test_text = "This is a test sentence to check if the model is working."
    try:
        result = paraphraser(test_text, max_length=50)
        return jsonify({
            "status": "Model loaded and working",
            "test_output": result[0]['generated_text'],
            "device": "GPU" if torch.cuda.is_available() else "CPU"
        })
    except Exception as e:
        return jsonify({
            "status": "Model loaded but not working",
            "error": str(e)
        }), 500

@app.route("/paraphrase", methods=["POST"])
def paraphrase_route():
    if not paraphraser:
        return jsonify({"error": "Paraphrasing service is currently unavailable. Model not loaded."}), 503
        
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided in JSON payload"}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Text field cannot be empty"}), 400
        
    logger.info(f"Received text for paraphrasing. Character count: {len(text)}")
    
    # Get multiple paraphrased versions
    result, status_code = safe_paraphrase(text, num_return_sequences=3)
    
    if status_code == 200:
        return jsonify({
            "paraphrases": result["paraphrases"],
            "original": text,
            "status": "success"
        }), 200
    
    return jsonify(result), status_code

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)