"""
Flask Web Application for House Price Prediction
Run with: python app/web_app.py
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
from model_inference import predict, load_model, load_metadata, VALID_LOCATIONS, VALID_PROPERTY_TYPES

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Load model once at startup
_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


@app.route('/')
def index():
    """Main prediction page."""
    metadata = load_metadata()
    return render_template('index.html',
                           locations=VALID_LOCATIONS,
                           property_types=VALID_PROPERTY_TYPES,
                           metadata=metadata)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for predictions."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'success': False, 'errors': ['No data provided']}), 400

        model = get_model()
        result = predict(data, model)

        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 422

    except Exception as e:
        return jsonify({'success': False, 'errors': [f'Server error: {str(e)}']}), 500


@app.route('/api/metadata', methods=['GET'])
def api_metadata():
    """Return model metadata and performance metrics."""
    metadata = load_metadata()
    # Remove predictions from response (large list)
    safe_metadata = {k: v for k, v in metadata.items() if k != 'predictions'}
    return jsonify(safe_metadata)


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        get_model()
        return jsonify({'status': 'ok', 'model': 'loaded'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed'}), 405


if __name__ == '__main__':
    print("🏠 House Price Prediction Web App")
    print("   URL: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
