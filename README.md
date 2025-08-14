# Runtime & Memory Prediction API

This project provides a **FastAPI backend** for training and predicting **EDA tool runtime and memory usage** from stage-level flow data stored in PostgreSQL.

It supports:
- **Training** two neural network models (runtime & memory) from historical logs.
- **Predicting** runtime and memory for new stage data.
- **Saving** predictions to CSV and optionally to an output database.

---

## 📂 Project Structure
.
├── runtime_backend.py # Main FastAPI backend
├── models/ # Saved models, scalers, encoders, metadata
├── static/ # Frontend files (index.html, etc.)
└── requirements.txt # Python dependencies

yaml
Copy
Edit

---

## ⚙️ Requirements
- Python 3.8+
- PostgreSQL
- The following Python packages:
```bash
pip install fastapi uvicorn sqlalchemy pandas numpy joblib tensorflow scikit-learn psycopg2
🗄 Database Setup
The backend connects to two PostgreSQL databases:

Input DB (runtime): stores your stage-level runtime/memory data.

Output DB (runtime_output): optional, for storing predictions.

Edit these in runtime_backend.py:

python
Copy
Edit
DB_URL = "postgresql://username:password@localhost:5432/runtime"
OUTPUT_DB_URL = "postgresql://username:password@localhost:5432/runtime_output"
🚀 Running the Server
bash
Copy
Edit
python runtime_backend.py
The API will be available at:

cpp
Copy
Edit
http://127.0.0.1:9321
📌 API Endpoints
1️⃣ Health Check
GET /health
Returns {"status": "ok"} to confirm the server is running.

2️⃣ Train Models
POST /train
Body:

json
Copy
Edit
{
  "table": "your_input_table"
}
What it does:

Detects relevant columns (block, stage, runtime, memory)

Parses runtime (HH:MM:SS → seconds) and memory (XM → MB)

Encodes categorical features, scales numeric features

Trains two Dense Neural Networks (runtime & memory)

Saves models, scalers, encoders, and metadata

Response Example:

json
Copy
Edit
{
  "message": "✅ Model trained for my_table",
  "table": "my_table",
  "r2_rt": 0.92,
  "r2_mb": 0.89,
  "calib_rt": 1.05,
  "calib_mb": 1.03
}
3️⃣ Predict Runtime & Memory
POST /predict
Body:

json
Copy
Edit
{
  "table": "your_input_table"
}
What it does:

Loads trained models & preprocessors

Applies same feature processing as training

Predicts runtime & memory

Formats runtime (HH:MM:SS) and memory (XM)

Saves results to CSV and output DB (if configured)

Response Example:

json
Copy
Edit
{
  "message": "✅ Prediction successful",
  "table": "my_table",
  "rows": 100,
  "file": "my_table_pred_20250101_120000.csv",
  "preview": [
    {"block": "blk1", "stage": "place", "predicted_runtime": "00:10:35", "predicted_memory": "512.3M"},
    {"block": "blk1", "stage": "cts",   "predicted_runtime": "00:12:15", "predicted_memory": "600.0M"}
  ]
}
4️⃣ Download Prediction CSV
GET /download/{filename}
Downloads the CSV file created in /predict.

