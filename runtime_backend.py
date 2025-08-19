# runtime_backend.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine
import pandas as pd, numpy as np, joblib, os, datetime, re
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score

# ---------- CONFIG ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DB_URL = "postgresql://postgres:Welcom%40123@localhost:5432/runtime"
OUTPUT_DB_URL = "postgresql://postgres:Welcom%40123@localhost:5432/runtime_outout"

engine = create_engine(DB_URL)
try:
    output_engine = create_engine(OUTPUT_DB_URL)
    _ = output_engine.connect()
    HAS_OUTPUT_DB = True
except Exception:
    HAS_OUTPUT_DB = False

app.mount("/static", StaticFiles(directory="static"), name="static")

os.makedirs("models", exist_ok=True)
RUNTIME_MODEL = "models/runtime_model.keras"
MEMORY_MODEL  = "models/memory_model.keras"
CAT_ENCODER   = "models/cat_encoder.save"
NUM_SCALER    = "models/num_scaler.save"
YSCALER_RT    = "models/y_scaler_rt.save"
YSCALER_MB    = "models/y_scaler_mb.save"
META_PATH     = "models/meta.save"

# ---------- API models ----------
class TrainRequest(BaseModel):
    table: str

class PredictRequest(BaseModel):
    table: str

# ---------- utilities ----------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def hhmmss_to_seconds(s: str):
    try:
        h,m,s = map(int, str(s).split(":"))
        return h*3600+m*60+s
    except:
        return None

def seconds_to_hhmmss(x: float):
    x = max(0, int(round(x)))
    h = x//3600; m=(x%3600)//60; s=x%60
    return f"{h:02d}:{m:02d}:{s:02d}"

def parse_memory_m(s):
    try:
        return float(str(s).replace("M","").strip())
    except:
        return None

def format_memory_m(x: float):
    return f"{float(x):.1f}M"

def load_table(table: str) -> pd.DataFrame:
    try:
        return pd.read_sql(f'SELECT * FROM "{table}"', engine)
    except:
        return pd.read_sql(f"SELECT * FROM {table}", engine)

def autodetect_cols(df: pd.DataFrame):
    block_col = next((c for c in df.columns if c.lower()=="block"), None)
    stage_col = next((c for c in df.columns if c.lower()=="stage"), None)
    time_col  = next((c for c in df.columns if c.lower() in ["runtime","real_time","cpu_time"]), None)
    mem_col   = next((c for c in df.columns if c.lower() in ["memory","peak_mem","peak_memory"]), None)
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
    set_seed(42)
    df = load_table(req.table)
    if df.empty:
        raise HTTPException(400,"Input table empty")

    block_col, stage_col, time_col, mem_col = autodetect_cols(df)
    if not (block_col and stage_col and time_col and mem_col):
        raise HTTPException(400,"Missing required columns")

    df["__runtime_sec"] = df[time_col].apply(hhmmss_to_seconds)
    df["__memory_mb"]  = df[mem_col].apply(parse_memory_m)
    df_labels = df.dropna(subset=["__runtime_sec","__memory_mb"]).copy()

    # categorical encoding
    cat_df = df_labels[[block_col,stage_col]].astype(str)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = enc.fit_transform(cat_df)

    # numeric features
    numeric = df_labels.select_dtypes(include=[np.number]).drop(columns=["__runtime_sec","__memory_mb"], errors='ignore')
    if not numeric.empty:
        num_scaler = StandardScaler()
        X_num = num_scaler.fit_transform(numeric)
        X = np.hstack([X_cat,X_num])
    else:
        num_scaler=None; X=X_cat

    y_rt = df_labels["__runtime_sec"].values.reshape(-1,1)
    y_mb = df_labels["__memory_mb"].values.reshape(-1,1)
    yscaler_rt = StandardScaler(); y_rt_s = yscaler_rt.fit_transform(y_rt)
    yscaler_mb = StandardScaler(); y_mb_s = yscaler_mb.fit_transform(y_mb)

    rt_model = build_dense(X.shape[1]); rt_model.fit(X,y_rt_s,epochs=300,batch_size=8,verbose=0)
    mb_model = build_dense(X.shape[1]); mb_model.fit(X,y_mb_s,epochs=300,batch_size=8,verbose=0)

    rt_model.save(RUNTIME_MODEL); mb_model.save(MEMORY_MODEL)
    joblib.dump(enc,CAT_ENCODER); joblib.dump(num_scaler,NUM_SCALER)
    joblib.dump(yscaler_rt,YSCALER_RT); joblib.dump(yscaler_mb,YSCALER_MB)
    joblib.dump({"block_col":block_col,"stage_col":stage_col},META_PATH)

    return {"message":"✅ Model trained","table":req.table}

@app.post("/predict")
def predict(req: PredictRequest):
    if not os.path.exists(RUNTIME_MODEL):
        raise HTTPException(400,"Train first")

    rt_model = load_model(RUNTIME_MODEL)
    mb_model = load_model(MEMORY_MODEL)
    enc = joblib.load(CAT_ENCODER)
    num_scaler = joblib.load(NUM_SCALER)
    yscaler_rt = joblib.load(YSCALER_RT)
    yscaler_mb = joblib.load(YSCALER_MB)
    meta = joblib.load(META_PATH)

    df = load_table(req.table)
    if df.empty: raise HTTPException(400,"Input empty")
    block_col, stage_col = meta["block_col"], meta["stage_col"]

    # Pick representative Block & Stage
    block = df[block_col].mode().iloc[0]
    stage = df[stage_col].mode().iloc[0]

    cat_df = pd.DataFrame({"block":[block],"stage":[stage]})
    X_cat = enc.transform(cat_df)

    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty and num_scaler is not None:
        X_num = num_scaler.transform([numeric.mean().values])
        X = np.hstack([X_cat,X_num])
    else:
        X=X_cat

    prt = yscaler_rt.inverse_transform(rt_model.predict(X)).reshape(-1)[0]
    pmb = yscaler_mb.inverse_transform(mb_model.predict(X)).reshape(-1)[0]

    result = pd.DataFrame([{
        "block":block,"stage":stage,
        "predicted_runtime":seconds_to_hhmmss(prt),
        "predicted_memory":format_memory_m(pmb)
    }])

    ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname=f"{req.table}_pred_{ts}.csv"; result.to_csv(fname,index=False)
    if HAS_OUTPUT_DB:
        try: result.to_sql(f"pred_{req.table}_{ts}",output_engine,index=False,if_exists="replace")
        except: pass

    return {"message":"✅ Prediction successful","preview":result.to_dict(orient="records"),"file":fname}

@app.get("/download/{filename}")
def download(filename:str):
    if not os.path.exists(filename):
        raise HTTPException(404,"File not found")
    return FileResponse(filename, media_type="text/csv", filename=filename)

@app.get("/health")
def health(): return {"status":"ok"}

if __name__=="__main__":
    import uvicorn
    uvicorn.run("runtime_backend:app",host="127.0.0.1",port=9321,reload=True)
