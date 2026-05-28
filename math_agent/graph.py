"""
graph.py — LangChain + LangGraph 에이전트 핵심 모듈
=====================================================
server.py와 lab.ipynb 양쪽에서 import해서 사용합니다.

아키텍처:
    START
      │
      ▼ (Conditional Edge: mode_router)
  ┌───────┐       ┌──────────────────┐
  │  chat │       │   calculator     │ ◀──┐
  │  node │       │   node           │    │
  └───┬───┘       └────────┬─────────┘    │
      │                    │ (should_continue)
      │              ┌─────▼──────┐       │
      │              │ tool_executor│──────┘
      │              └─────────────┘
      ▼                    ▼
     END                  END
"""

import ast
import os
from typing import Annotated, Literal, TypedDict

import numpy as np
import sympy as sp
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from proxy_config import make_llm

# ─────────────────────────────────────────────────────────────
# Phoenix 연동
# ─────────────────────────────────────────────────────────────

# 폴백 프롬프트 (Phoenix 연결 실패 시 사용)
_FALLBACK_PROMPTS = {
    "chat_system": (
        "당신은 친절하고 유능한 AI 어시스턴트입니다.\n"
        "사용자의 질문에 명확하고 도움이 되는 답변을 제공하세요.\n"
        "답변은 한국어로 작성하세요."
    ),
    "calculator_system": (
        "당신은 수학 계산 전문가입니다.\n"
        "사용자의 계산 요청을 분석하고 반드시 도구를 사용해 계산하세요.\n"
        "머릿속으로 계산하지 말고 반드시 도구를 호출하세요."
    ),
}


def pull_prompt(name: str) -> str:
    """
    Phoenix Prompt Hub에서 최신 버전 프롬프트를 가져옵니다.
    연결 실패 시 폴백 프롬프트를 반환합니다.
    """
    try:
        from phoenix.client import Client
        client = Client()
        p = client.prompts.get(prompt_identifier=name)
        # PromptVersion 내부 _template["messages"] 에서 system 내용을 추출
        msgs = p._template.get("messages", [])
        if msgs:
            content = msgs[0].get("content", "")
            if isinstance(content, str):
                return content
            # content가 리스트(TextContentPart 등)인 경우
            return " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return _FALLBACK_PROMPTS.get(name, "You are a helpful assistant.")
    except Exception as e:
        print(f"[Phoenix] '{name}' 프롬프트 로드 실패: {e}")
        print(f"[Fallback] 기본 프롬프트를 사용합니다.")
        return _FALLBACK_PROMPTS.get(name, "You are a helpful assistant.")


# ─────────────────────────────────────────────────────────────
# State 정의
# ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # 대화 히스토리 (자동 누적)
    mode: str                                              # 현재 모드: "chat" | "calculator"


_MSG_TYPE_MAP = {"human": HumanMessage, "ai": AIMessage, "system": SystemMessage, "tool": ToolMessage}


def _coerce_messages(messages: list) -> list[BaseMessage]:
    """LangServe 역직렬화 시 BaseMessage이지만 HumanMessage가 아닌 중간 타입으로
    역직렬화되는 경우를 보정. isinstance 대신 type 필드 기준으로 재생성."""
    result = []
    for m in messages:
        t = getattr(m, "type", None)
        cls = _MSG_TYPE_MAP.get(t)
        if cls is not None and not isinstance(m, cls):
            if cls is ToolMessage:
                result.append(cls(
                    content=getattr(m, "content", ""),
                    tool_call_id=getattr(m, "tool_call_id", ""),
                ))
            else:
                result.append(cls(content=getattr(m, "content", "")))
        else:
            result.append(m)
    return result


# ─────────────────────────────────────────────────────────────
# Tools 정의
# ─────────────────────────────────────────────────────────────

@tool
def arithmetic(expression: str) -> str:
    """
    사칙연산을 안전하게 계산합니다.
    지원: +, -, *, /, **, % (거듭제곱, 나머지)
    예: '23 * 47 + 15', '(100 - 37) * 4 / 2', '2 ** 10'
    """
    try:
        # ast.parse로 안전하게 파싱 (eval 직접 사용 금지)
        tree = ast.parse(expression.strip(), mode="eval")
        # 허용된 노드 타입만 통과
        allowed = {
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
            ast.Mod, ast.Pow, ast.USub, ast.UAdd,
        }
        for node in ast.walk(tree):
            if type(node) not in allowed:
                return f"오류: 허용되지 않는 연산입니다 ({type(node).__name__})"
        result = eval(compile(tree, "<string>", "eval"))
        return str(result)
    except Exception as e:
        return f"계산 오류: {e}"


@tool
def calculus(expression: str, variable: str = "x", operation: str = "diff") -> str:
    """
    미분 또는 적분을 계산합니다 (sympy 기반).
    - expression: 수식 문자열. 예: 'x**3 + 2*x', 'sin(x)', 'x**2 * exp(x)'
    - variable: 미분/적분 변수. 기본값: 'x'
    - operation: 'diff'(미분) 또는 'integrate'(적분)
    """
    try:
        var = sp.Symbol(variable)
        expr = sp.sympify(expression)
        if operation == "diff":
            result = sp.diff(expr, var)
            return str(result)
        elif operation == "integrate":
            result = sp.integrate(expr, var)
            return str(result)
        else:
            return f"오류: 지원하지 않는 연산 '{operation}'. 'diff' 또는 'integrate'를 사용하세요."
    except Exception as e:
        return f"계산 오류: {e}"


@tool
def matrix_calc(
    matrix_a: list,
    matrix_b: list = None,
    operation: str = "det",
) -> str:
    """
    행렬 연산을 수행합니다 (numpy 기반).
    - matrix_a: 행렬 A (2차원 리스트). 예: [[1,2],[3,4]]
    - matrix_b: 행렬 B (matmul 연산에만 필요)
    - operation: 'det'(행렬식), 'inv'(역행렬), 'matmul'(행렬곱)
    """
    try:
        A = np.array(matrix_a, dtype=float)
        if operation == "det":
            result = np.linalg.det(A)
            return str(round(result, 6))
        elif operation == "inv":
            result = np.linalg.inv(A)
            return str(result.tolist())
        elif operation == "matmul":
            if matrix_b is None:
                return "오류: matmul 연산에는 matrix_b가 필요합니다."
            B = np.array(matrix_b, dtype=float)
            result = np.matmul(A, B)
            return str(result.tolist())
        else:
            return f"오류: 지원하지 않는 연산 '{operation}'. 'det', 'inv', 'matmul' 중 하나를 사용하세요."
    except np.linalg.LinAlgError as e:
        return f"행렬 오류: {e} (역행렬이 존재하지 않을 수 있습니다)"
    except Exception as e:
        return f"계산 오류: {e}"


tools = [arithmetic, calculus, matrix_calc]


# ─────────────────────────────────────────────────────────────
# LLM 설정
# ─────────────────────────────────────────────────────────────

llm = make_llm(model="claude-haiku-4-5-20251001", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# ─────────────────────────────────────────────────────────────
# Mode Router (LLM Classifier)
# ─────────────────────────────────────────────────────────────

class ModeDecision(BaseModel):
    """모드 분류 결과"""
    mode: Literal["chat", "calculator"]
    reason: str  # 분류 근거 (디버깅용)


_classifier_llm = make_llm(
    model="claude-haiku-4-5-20251001",
    temperature=0,
).with_structured_output(ModeDecision)

_CLASSIFIER_SYSTEM = """사용자 메시지를 다음 두 모드 중 하나로 분류하세요.

calculator: 수학적 계산이 필요한 경우
  - 사칙연산, 수식 계산
  - 미분, 적분
  - 행렬 연산 (행렬식, 역행렬, 행렬곱)
  예: "23 * 47 계산해줘", "x^3을 미분해줘", "역행렬 구해줘"

chat: 그 외 모든 경우
  - 일반 대화, 설명 요청
  - 계산 방법 설명 (계산 자체를 요구하지 않음)
  - 감사 인사, 질문
  예: "미분이 뭔가요?", "안녕하세요", "도움 고맙습니다"

주의: "미분 계산 설명해줘"는 계산을 요구하는 것이 아니므로 chat입니다."""


def mode_router(state: AgentState) -> Literal["chat", "calculator"]:
    """마지막 메시지를 분석해 라우팅 키를 반환합니다."""
    if not state["messages"]:
        return "chat"
    last_msg = state["messages"][-1]
    result = _classifier_llm.invoke([
        SystemMessage(content=_CLASSIFIER_SYSTEM),
        HumanMessage(content=last_msg.content),
    ])
    print(f"[Router] mode={result.mode}, reason={result.reason}")
    return result.mode


# ─────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────

def chat_node(state: AgentState) -> dict:
    """일반 대화 노드: Phoenix에서 프롬프트 Pull → LLM 호출"""
    system_prompt = pull_prompt("chat_system")
    messages = [SystemMessage(content=system_prompt)] + _coerce_messages(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response], "mode": "chat"}


def calculator_node(state: AgentState) -> dict:
    """계산기 노드: Phoenix에서 프롬프트 Pull → LLM + Tools 호출"""
    system_prompt = pull_prompt("calculator_system")
    messages = [SystemMessage(content=system_prompt)] + _coerce_messages(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response], "mode": "calculator"}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Tool call이 있으면 tool_executor로, 없으면 종료"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "end"


tool_executor = ToolNode(tools)


# ─────────────────────────────────────────────────────────────
# Graph 조립
# ─────────────────────────────────────────────────────────────

builder = StateGraph(AgentState)

# 노드 등록
builder.add_node("chat_node", chat_node)
builder.add_node("calculator_node", calculator_node)
builder.add_node("tool_executor", tool_executor)

# 엣지 연결
builder.add_conditional_edges(
    START,
    mode_router,
    {"chat": "chat_node", "calculator": "calculator_node"},
)
builder.add_edge("chat_node", END)
builder.add_conditional_edges(
    "calculator_node",
    should_continue,
    {"tools": "tool_executor", "end": END},
)
builder.add_edge("tool_executor", "calculator_node")

# 컴파일 (서버 및 노트북에서 import)
graph = builder.compile()
