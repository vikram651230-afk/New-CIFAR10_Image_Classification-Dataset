import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")

    print("=" * 65)
    print("  CIFAR-10 Pure-ANN Image Classification Flask Web Service")
    print(f"  URL: http://127.0.0.1:{port}")
    print("=" * 65)

    app.run(host=host, port=port, debug=debug)
