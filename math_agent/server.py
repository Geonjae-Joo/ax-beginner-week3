"""
server.py — LangServe FastAPI 서버
=====================================
사전 준비:
    1. 터미널에서 phoenix serve 실행 → http://localhost:6006
    2. python server.py 또는 uvicorn server:app --reload --port 8000

엔드포인트 (add_routes 자동 생성):
    /agent/invoke       ← 단일 요청 동기
    /agent/batch        ← 다중 요청 병렬
    /agent/stream       ← SSE 토큰 스트리밍
    /agent/stream_log   ← 중간 단계 포함 스트리밍
    /agent/docs         ← Swagger 자동 문서
    /agent/playground   ← 브라우저 테스트 UI [보조]

참고:
    LangServe = 수업·로컬 데모용.
    프로덕션 배포는 LangGraph Platform(langgraph-cli) 사용.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

# Phoenix Tracing 연결 (phoenix serve가 먼저 실행된 상태여야 함)
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

tracer_provider = register(
    project_name="math-agent",
    endpoint="http://localhost:6006/v1/traces",
)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# graph.py에서 컴파일된 그래프 import
from graph import graph, AgentState

# ─── FastAPI 앱 설정 ──────────────────────────────────────────
app = FastAPI(
    title="Math Agent — LangServe",
    description="LangChain + LangGraph + Phoenix 수업 데모 서버",
    version="1.0.0",
)

# CORS (playground 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── LangServe 라우트 등록 ────────────────────────────────────
add_routes(
    app,
    graph.with_types(input_type=AgentState),
    path="/agent",
)


@app.get("/")
def root():
    return {
        "endpoints": {
            "/agent/invoke":      "단일 요청 동기",
            "/agent/batch":       "다중 요청 병렬",
            "/agent/stream":      "SSE 스트리밍",
            "/agent/stream_log":  "중간 단계 스트리밍",
            "/agent/docs":        "Swagger 문서",
            "/agent/playground":  "브라우저 UI [보조]",
        },
        "phoenix": "http://localhost:6006",
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 서버 시작: http://localhost:8000")
    print("📡 /invoke  /batch  /stream  /stream_log  /docs")
    print("🎮 /agent/playground  [브라우저 테스트 UI]")
    print("📊 Phoenix: http://localhost:6006")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
