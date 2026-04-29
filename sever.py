from app import app

print("🚀 Server Starting...")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)