# QuantumSentinel — Config & Backend Tasks

## Phase 1: Configuration File
- [x] Create `config.py` at project root

## Phase 2: Backend — Foundation
- [x] Create `backend/__init__.py`
- [x] Create `backend/schemas.py` (Pydantic models)
- [x] Create `backend/main.py` (FastAPI app + health endpoint)

## Phase 3: Backend — Routers
- [x] Create `backend/routers/__init__.py`
- [x] Create `backend/routers/stocks.py`
- [x] Create `backend/routers/sentiment.py`
- [x] Create `backend/routers/predictions.py`

## Phase 4: Refactor Source Modules
- [x] Update `src/data_collector.py` to use config
- [x] Update `src/sentiment_analyzer.py` to use config
- [x] Update `src/predictor.py` to use config
- [x] Update `src/train.py` to use config

## Phase 5: Supporting Files
- [x] Update `requirements.txt` (add fastapi, uvicorn, pydantic)
- [x] Create `run_backend.bat`

## Phase 6: Verify
- [x] Verify config imports ✅
- [x] Verify backend imports ✅
- [x] Server started successfully — 14 routes confirmed via OpenAPI ✅
- [x] Health endpoint responded 200 OK ✅
