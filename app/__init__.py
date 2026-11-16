import os
from flask import Flask, jsonify
from app.config import Config

def create_app(config_class=Config):
    """Flask application factory."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure required directories exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["MODELS_DIR"], exist_ok=True)

    # Register blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    # Error handlers
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            "success": False,
            "error": "Bad Request",
            "message": getattr(error, "description", "The request could not be understood."),
        }), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "success": False,
            "error": "Not Found",
            "message": "The requested URL or resource was not found.",
        }), 404

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            "success": False,
            "error": "File Too Large",
            "message": "The uploaded file exceeds the 16MB size limit.",
        }), 413

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "success": False,
            "error": "Internal Server Error",
            "message": "An unexpected server error occurred.",
        }), 500

    return app
