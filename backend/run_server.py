from app import app, db

if __name__ == "__main__":
    with app.app_context():
        try:
            db.create_all()
            print("Database tables created successfully.")
        except Exception as e:
            print(f"Database initialization error: {e}")

    app.run(debug=True, host="0.0.0.0", port=5000)
