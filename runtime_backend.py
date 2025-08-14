# runtime_backend.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine
import pandas as pd, numpy as np, joblib, os, datetime, re
import random
from typing import List, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ---------- CONFIG ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# input DB (tables with your stage-level rows)
DB_URL = "postgresql://postgres:Welcom%40123@localhost:5432/runtime"
# output DB for predictions (must exist)
OUTPUT_DB_URL = "postgresql://postgres:Welcom%40123@localhost:5432/runtime_outout"

engine = create_engine(DB_URL)
try:
    output_engine = create_engine(OUTPUT_DB_URL)
    _ = output_engine.connect()
    HAS_OUTPUT_DB = True
except Exception:
    HAS_OUTPUT_DB = False

app.mount("/static", StaticFiles(directory="static"), name="static")

# artifact paths
os.makedirs("models", exist_ok=True)
RUNTIME_MODEL = "models/runtime_model.keras"
MEMORY_MODEL  = "models/memory_model.keras"
CAT_ENCODER   = "models/cat_encoder.save"
NUM_SCALER    = "models/num_scaler.save"
YSCALER_RT    = "models/y_scaler_rt.save"
YSCALER_MB    = "models/y_scaler_mb.save"
META_PATH     = "models/meta.save"        # stores detected column names and calibration multipliers

# ---------- API models ----------
class TrainRequest(BaseModel):
    table: str

class PredictRequest(BaseModel):
    table: str

# ---------- utilities ----------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def hhmmss_to_seconds(s: str) -> Optional[int]:
    if not isinstance(s, str): return None
    m = re.match(r"^\s*(\d{1,2}):(\d{2}):(\d{2})\s*$", s)
    if not m: return None
    h, mm, ss = map(int, m.groups()); return h*3600 + mm*60 + ss

def seconds_to_hhmmss(x: float) -> str:
    x = max(0, int(round(x)))
    h = x//3600; m = (x%3600)//60; s = x%60
    return f"{h:02d}:{m:02d}:{s:02d}"

def parse_memory_m(s: str) -> Optional[float]:
    if isinstance(s, (int, float)): return float(s)
    if not isinstance(s, str): return None
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*M\s*$", s, re.IGNORECASE)
    return float(m.group(1)) if m else None

def format_memory_m(x: float) -> str:
    return f"{float(x):.1f}M"

def load_table(table: str) -> pd.DataFrame:
    # tolerant quoting
    try:
        return pd.read_sql(f'SELECT * FROM "{table}"', engine)
    except Exception:
        return pd.read_sql(f"SELECT * FROM {table}", engine)

def autodetect_cols(df: pd.DataFrame):
    # stage, block, time, memory detection
    stage_col = None
    for c in df.columns:
        if c.lower() == "stage":
            stage_col = c; break
    if stage_col is None:
        # guess by values
        for c in df.columns:
            vals = set(str(v).strip().lower() for v in df[c].dropna().unique()[:50])
            if vals & {"place", "cts", "route", "floorplan", "synth"}:
                stage_col = c; break
    block_col = next((c for c in df.columns if c.lower()=="block"), None)
    time_col = next((c for c in df.columns if c.lower() in ["real_time","runtime","run_time","cpu_time"]), None)
    mem_col  = next((c for c in df.columns if c.lower() in ["memory","peak_mem","peak_memory"]), None)
    return block_col, stage_col, time_col, mem_col

def build_dense(input_dim:int):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),
        Dense(64, activation='relu'),
        Dropout(0.15),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.002), loss='mse')
    return model

# ---------- endpoints ----------
@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("static/index.html","r",encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return HTMLResponse(content=f"frontend missing: {e}", status_code=500)

@app.post("/train")
def train(req: TrainRequest):
    """
    Trains runtime and memory regressors using real labels from the given table.
    """
    try:
        set_seed(42)
        df = load_table(req.table)
        if df.empty:
            raise HTTPException(status_code=400, detail="Input table empty")

        block_col, stage_col, time_col, mem_col = autodetect_cols(df)
        if stage_col is None or time_col is None or mem_col is None:
            raise HTTPException(status_code=400, detail=f"Could not detect required columns (stage/time/memory). Detected: stage={stage_col}, time={time_col}, memory={mem_col}")

        # parse labels
        df["__runtime_sec"] = df[time_col].apply(hhmmss_to_seconds)
        df["__memory_mb"]  = df[mem_col].apply(parse_memory_m)
        df_labels = df.dropna(subset=["__runtime_sec","__memory_mb"]).copy()
        if df_labels.shape[0] < 4:
            raise HTTPException(status_code=400, detail="Not enough rows with valid runtime+memory to train (>=4 required)")

        # categorical features: block + stage (one-hot)
        blocks = df_labels[block_col] if block_col and block_col in df_labels.columns else pd.Series([req.table.split("_")[0]]*len(df_labels))
        stages = df_labels[stage_col].astype(str)
        cat_df = pd.DataFrame({"block": blocks, "stage": stages})

        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = enc.fit_transform(cat_df)

        # numeric features — include any numeric columns except labels
        numeric = df_labels.select_dtypes(include=[np.number]).drop(columns=["__runtime_sec","__memory_mb"], errors='ignore')
        if not numeric.empty:
            num_scaler = StandardScaler()
            X_num = num_scaler.fit_transform(numeric)
            X = np.hstack([X_cat, X_num])
        else:
            num_scaler = None
            X = X_cat

        # targets and scalers
        y_rt = df_labels["__runtime_sec"].values.reshape(-1,1)
        y_mb = df_labels["__memory_mb"].values.reshape(-1,1)
        yscaler_rt = StandardScaler(); y_rt_s = yscaler_rt.fit_transform(y_rt)
        yscaler_mb = StandardScaler(); y_mb_s = yscaler_mb.fit_transform(y_mb)

        # Train runtime model
        rt_model = build_dense(X.shape[1])
        rt_model.fit(X, y_rt_s, epochs=400, batch_size=8, validation_split=0.15, verbose=0)

        # Train memory model
        mb_model = build_dense(X.shape[1])
        mb_model.fit(X, y_mb_s, epochs=400, batch_size=8, validation_split=0.15, verbose=0)

        # Evaluate on train to compute calibration multiplier
        pred_rt_s = rt_model.predict(X, verbose=0)
        pred_mb_s = mb_model.predict(X, verbose=0)
        pred_rt = yscaler_rt.inverse_transform(pred_rt_s).reshape(-1)
        pred_mb = yscaler_mb.inverse_transform(pred_mb_s).reshape(-1)

        # compute calibration multiplier so predictions are not lower than actual for most rows
        # multiplier = max(1.0, 90th percentile of (actual / predicted))
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_rt = (y_rt.reshape(-1) / (pred_rt + 1e-9))
            ratio_mb = (y_mb.reshape(-1) / (pred_mb + 1e-9))
        # handle that predictions might be zero or negative; filter finite positive
        ratio_rt_valid = ratio_rt[np.isfinite(ratio_rt) & (ratio_rt>0)]
        ratio_mb_valid = ratio_mb[np.isfinite(ratio_mb) & (ratio_mb>0)]
        if len(ratio_rt_valid)>0:
            mult_rt = max(1.0, float(np.percentile(ratio_rt_valid, 90)))
        else:
            mult_rt = 1.0
        if len(ratio_mb_valid)>0:
            mult_mb = max(1.0, float(np.percentile(ratio_mb_valid, 90)))
        else:
            mult_mb = 1.0

        # Save artifacts
        rt_model.save(RUNTIME_MODEL)
        mb_model.save(MEMORY_MODEL)
        joblib.dump(enc, CAT_ENCODER)
        joblib.dump(num_scaler, NUM_SCALER)
        joblib.dump(yscaler_rt, YSCALER_RT)
        joblib.dump(yscaler_mb, YSCALER_MB)
        joblib.dump({
            "block_col": block_col,
            "stage_col": stage_col,
            "time_col": time_col,
            "mem_col": mem_col,
            "calib_rt": mult_rt,
            "calib_mb": mult_mb
        }, META_PATH)

        # report r2 on train for info
        r2_rt = r2_score(y_rt.reshape(-1), pred_rt) if len(pred_rt)>1 else float("nan")
        r2_mb = r2_score(y_mb.reshape(-1), pred_mb) if len(pred_mb)>1 else float("nan")

        return {"message": f"✅ Model trained for {req.table}", "table": req.table, "r2_rt": float(r2_rt), "r2_mb": float(r2_mb), "calib_rt": mult_rt, "calib_mb": mult_mb}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        # load artifacts
        rt_model = load_model(RUNTIME_MODEL)
        mb_model = load_model(MEMORY_MODEL)
        enc = joblib.load(CAT_ENCODER)
        num_scaler = joblib.load(NUM_SCALER)
        yscaler_rt = joblib.load(YSCALER_RT)
        yscaler_mb = joblib.load(YSCALER_MB)
        meta = joblib.load(META_PATH)

        df = load_table(req.table)
        if df.empty:
            raise HTTPException(status_code=400, detail="Input table empty")

        block_col = meta.get("block_col")
        stage_col = meta["stage_col"]
        if stage_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Stage column '{stage_col}' not present in table")

        # build categorical frame
        blocks = df[block_col] if block_col and block_col in df.columns else pd.Series([req.table.split("_")[0]]*len(df))
        stages = df[stage_col].astype(str)
        cat_df = pd.DataFrame({"block": blocks, "stage": stages})
        X_cat = enc.transform(cat_df)

        # numeric
        numeric = df.select_dtypes(include=[np.number])
        if not numeric.empty and num_scaler is not None:
            X_num = num_scaler.transform(numeric)
            X = np.hstack([X_cat, X_num])
        else:
            X = X_cat

        # predict and inverse-scale
        pred_rt_s = rt_model.predict(X, verbose=0)
        pred_mb_s = mb_model.predict(X, verbose=0)
        pred_rt = yscaler_rt.inverse_transform(pred_rt_s).reshape(-1)
        pred_mb = yscaler_mb.inverse_transform(pred_mb_s).reshape(-1)

        # calibration multipliers
        mult_rt = float(meta.get("calib_rt", 1.0))
        mult_mb = float(meta.get("calib_mb", 1.0))
        pred_rt = pred_rt * mult_rt
        pred_mb = pred_mb * mult_mb

        # format results
        out = []
        for i in range(len(pred_rt)):
            block_val = str(blocks.iloc[i]) if i < len(blocks) else req.table.split("_")[0]
            stage_val = str(stages.iloc[i])
            prt = seconds_to_hhmmss(float(pred_rt[i]))
            pmb = format_memory_m(float(pred_mb[i]))
            out.append({"block": block_val, "stage": stage_val, "predicted_runtime": prt, "predicted_memory": pmb})

        results = pd.DataFrame(out)

        # save CSV & to output db if possible
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{req.table}_pred_{ts}.csv"
        results.to_csv(fname, index=False)
        if HAS_OUTPUT_DB:
            try:
                tbl = f"pred_{req.table}_{ts}"
                results.to_sql(tbl, output_engine, index=False, if_exists="replace")
            except Exception:
                pass

        return {"message": "✅ Prediction successful", "table": req.table, "rows": int(results.shape[0]), "file": fname, "preview": results.to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
def download(filename: str):
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filename, media_type="text/csv", filename=filename)

@app.get("/health")
def health():
    return {"status":"ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("runtime_backend:app", host="127.0.0.1", port=9321, reload=True)
