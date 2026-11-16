import os
from werkzeug.utils import secure_filename

def allowed_file(filename, allowed_extensions):
    """Check if uploaded file has an allowed extension."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in allowed_extensions

def sanitize_filename(filename):
    """Sanitize filename to prevent directory traversal attacks."""
    return secure_filename(filename)
