# Math Agent — LangServe

LangGraph로 구성한 수학 계산 에이전트를 LangServe(FastAPI)로 서빙하는 데모 서버입니다.  
Phoenix로 트레이싱을 연동해 요청 흐름을 시각적으로 모니터링할 수 있습니다.

## 아키텍처

```
사용자 요청
    │
    ▼ (LLM Classifier: mode_router)
 ┌────────┐        ┌──────────────────┐
 │  chat  │        │   calculator     │ ◀──┐
 │  node  │        │   node           │    │
 └───┬────┘        └────────┬─────────┘    │
     │                      │ tool_calls?  │
     │               ┌──────▼──────┐       │
     │               │tool_executor│───────┘
     ▼               └─────────────┘
    END                    END
```

| 모드 | 설명 |
|---|---|
| `chat` | 일반 대화, 설명 요청 |
| `calculator` | 수식 계산, 미분/적분, 행렬 연산 |

### 도구 (Tools)

| 도구 | 설명 |
|---|---|
| `arithmetic` | 사칙연산, 거듭제곱, 나머지 (`ast` 기반 안전 파싱) |
| `calculus` | 미분·적분 (`sympy`) |
| `matrix_calc` | 행렬식·역행렬·행렬곱 (`numpy`) |

## 환경 설정

### 1. `.env` 파일 작성

```dotenv
ANTHROPIC_API_KEY=sk-ant-...

# 사내 프록시가 필요한 경우
USE_PROXY=true
PROXY_URL=http://proxy.corp.example.com:8080
```

프록시가 불필요하면 `USE_PROXY=false`로 설정하거나 해당 항목을 제거합니다.

### 2. 패키지 설치

```bash
pip install langchain langchain-anthropic langgraph langserve fastapi uvicorn
pip install numpy sympy
pip install arize-phoenix openinference-instrumentation-langchain
```

## 실행

### 1단계 — Phoenix 트레이싱 서버 시작

```bash
phoenix serve
```

Phoenix UI: http://localhost:6006

### 2단계 — LangServe 서버 시작

```bash
python server.py
# 또는
uvicorn server:app --reload --port 8000
```

서버 기동 후 http://localhost:8000 에서 사용 가능합니다.

## API 엔드포인트

| 엔드포인트 | 설명 |
|---|---|
| `GET /` | 엔드포인트 목록 |
| `POST /agent/invoke` | 단일 요청 (동기) |
| `POST /agent/batch` | 다중 요청 병렬 처리 |
| `POST /agent/stream` | SSE 토큰 스트리밍 |
| `POST /agent/stream_log` | 중간 단계 포함 스트리밍 |
| `GET /agent/docs` | Swagger 자동 문서 |
| `GET /agent/playground` | 브라우저 테스트 UI |

### 요청 예시

```bash
curl -X POST http://localhost:8000/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "human", "content": "23 * 47 계산해줘"}], "mode": ""}}'
```

## 참고

- LangServe는 수업·로컬 데모 용도입니다.
- 프로덕션 배포는 **LangGraph Platform** (`langgraph-cli`) 사용을 권장합니다.
