"""
proxy_config.py — 사내 프록시 중앙 설정 모듈
=============================================
.env 파일에서 USE_PROXY=true/false 로 ON/OFF 제어합니다.

.env 예시:
    USE_PROXY=true          # 프록시 사용 여부 (true/false)
    PROXY_URL=http://proxy.corp.example.com:8080
    ANTHROPIC_API_KEY=sk-ant-...

사용법:
    from proxy_config import make_llm, make_eval_model

    llm = make_llm()                                    # ChatAnthropic
    llm_with_tools = make_llm().bind_tools(tools)
    eval_model = make_eval_model()                      # phoenix.evals.LLM
"""

import os
import httpx
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

# ── 프록시 ON/OFF 설정 ──────────────────────────────────────────────────────
# .env 에서 USE_PROXY=true 로 활성화, false(기본값)로 비활성화
USE_PROXY: bool = os.getenv("USE_PROXY", "false").lower() in ("true", "1", "yes")
PROXY_URL: str  = os.getenv("PROXY_URL", "")

# localhost 는 프록시 우회
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

# ── 내부 헬퍼 ───────────────────────────────────────────────────────────────

def _sync_http() -> httpx.Client | None:
    if USE_PROXY and PROXY_URL:
        return httpx.Client(proxy=PROXY_URL, verify=False)
    return None


def _async_http() -> httpx.AsyncClient | None:
    if USE_PROXY and PROXY_URL:
        return httpx.AsyncClient(proxy=PROXY_URL, verify=False)
    return None


# ── 공개 팩토리 함수 ────────────────────────────────────────────────────────

def make_llm(
    model: str = "claude-haiku-4-5-20251001",
    temperature: float = 0,
    max_tokens: int = 1024,
):
    """프록시 설정이 적용된 ChatAnthropic 인스턴스를 반환합니다."""
    from langchain_anthropic import ChatAnthropic
    from anthropic import Anthropic, AsyncAnthropic

    llm = ChatAnthropic(model=model, temperature=temperature, max_tokens=max_tokens)
    if USE_PROXY and PROXY_URL:
        llm._client       = Anthropic(http_client=_sync_http())
        llm._async_client = AsyncAnthropic(http_client=_async_http())
    return llm


def make_anthropic_client():
    """프록시 설정이 적용된 Anthropic 동기 클라이언트를 반환합니다."""
    from anthropic import Anthropic
    http = _sync_http()
    return Anthropic(http_client=http) if http else Anthropic()


def make_async_anthropic_client():
    """프록시 설정이 적용된 Anthropic 비동기 클라이언트를 반환합니다."""
    from anthropic import AsyncAnthropic
    http = _async_http()
    return AsyncAnthropic(http_client=http) if http else AsyncAnthropic()


@contextmanager
def proxy_patched_anthropic():
    """
    내부적으로 Anthropic()을 직접 생성하는 라이브러리(phoenix.evals 등)에
    프록시 http_client를 자동 주입하는 컨텍스트 매니저입니다.

    사용 예:
        with proxy_patched_anthropic():
            eval_model = LLM(provider="anthropic", model="claude-haiku-4-5-20251001")
    """
    if not (USE_PROXY and PROXY_URL):
        yield
        return

    import anthropic as _anthropic
    _orig = _anthropic.Anthropic.__init__

    def _patched(self, *args, **kwargs):
        if "http_client" not in kwargs:
            kwargs["http_client"] = httpx.Client(proxy=PROXY_URL, verify=False)
        _orig(self, *args, **kwargs)

    _anthropic.Anthropic.__init__ = _patched
    try:
        yield
    finally:
        _anthropic.Anthropic.__init__ = _orig


def make_eval_model(model: str = "claude-haiku-4-5-20251001"):
    """프록시 설정이 적용된 phoenix.evals.LLM 인스턴스를 반환합니다."""
    from phoenix.evals import LLM
    with proxy_patched_anthropic():
        return LLM(provider="anthropic", model=model)


# ── 로드 확인 메시지 ────────────────────────────────────────────────────────
_status = f"USE_PROXY={USE_PROXY}"
if USE_PROXY:
    _status += f", PROXY_URL={PROXY_URL}"
print(f"[proxy_config] {_status}")
