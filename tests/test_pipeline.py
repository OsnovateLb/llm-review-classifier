"""Тесты пайплайна без интернета и ключа: поддельный клиент и локальный HTTP-сервер,
имитирующий OpenAI-совместимый API (OpenRouter)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import classify_reviews as cr


# --------------------------------------------------------------------------- поддельный клиент


class FakeClient:
    model = "fake/model:free"
    base_url = "http://fake"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def chat(self, messages, response_format, max_tokens=4000):
        self.calls.append({"messages": [dict(m) for m in messages], "response_format": response_format})
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
        return text, {"prompt_tokens": 500, "completion_tokens": 120}


def ok_payload(ids, sentiment="positive"):
    return {"results": [{"id": i, "sentiment": sentiment, "topic": "цены", "confidence": 0.9, "summary": "ок"} for i in ids]}


def bad_request():
    err = cr.LLMError("response_format not supported")
    err.status_code = 400
    return err


# --------------------------------------------------------------------------- разбор и проверка


def test_parse_valid_and_normalizes_case():
    raw = json.dumps({"results": [{"id": 1, "sentiment": "Positive", "topic": "Цены", "confidence": 1.4, "summary": " x "}]})
    assert cr.parse_and_validate(raw, [1]) == [{"id": 1, "sentiment": "positive", "topic": "цены", "confidence": 1.0, "summary": "x"}]


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"results": [{"id": 1, "sentiment": "neutral", "topic": "другое", "confidence": 0.5, "summary": "s"}]}\n```',
        '<think>размышляю…</think>{"results": [{"id": 1, "sentiment": "neutral", "topic": "другое", "confidence": 0.5, "summary": "s"}]}',
        'Вот ответ: {"results": [{"id": 1, "sentiment": "neutral", "topic": "другое", "confidence": 0.5, "summary": "s"}]} Готово.',
    ],
)
def test_extracts_json_from_noisy_answers(raw):
    assert cr.parse_and_validate(raw, [1])[0]["sentiment"] == "neutral"


@pytest.mark.parametrize(
    "raw, msg",
    [
        ("не json", "корректным JSON"),
        ('{"foo": []}', "results"),
        ('{"results": [{"id": 1, "sentiment": "angry", "topic": "цены", "confidence": 0.5, "summary": ""}]}', "тональность"),
        ('{"results": [{"id": 1, "sentiment": "neutral", "topic": "погода", "confidence": 0.5, "summary": ""}]}', "тема"),
        ('{"results": []}', "нет результатов"),
    ],
)
def test_parse_rejects_bad_output(raw, msg):
    with pytest.raises(cr.ValidationError, match=msg):
        cr.parse_and_validate(raw, [1])


# --------------------------------------------------------------------------- логика запросов


def test_retry_after_invalid_json():
    client = FakeClient(["oops", ok_payload([1, 2])])
    usage = cr.Counter()
    out = cr.classify_batch(client, [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}], usage)
    assert [o["id"] for o in out] == [1, 2]
    assert usage["requests"] == 2 and usage["prompt_tokens"] == 1000
    assert "не прошёл проверку" in client.calls[1]["messages"][-1]["content"]
    assert client.calls[0]["response_format"]["type"] == "json_schema"


def test_falls_back_through_response_formats():
    client = FakeClient([bad_request(), bad_request(), ok_payload([1])])
    out = cr.classify_batch(client, [{"id": 1, "text": "a"}], cr.Counter())
    assert out[0]["sentiment"] == "positive"
    assert [c["response_format"] and c["response_format"]["type"] for c in client.calls] == ["json_schema", "json_object", None]


def test_gives_up_after_max_attempts():
    client = FakeClient(["x", "x", "x"])
    with pytest.raises(RuntimeError):
        cr.classify_batch(client, [{"id": 1, "text": "a"}], cr.Counter())


def test_summary_agreement():
    items = [
        {"sentiment": "positive", "topic": "цены", "confidence": 1.0, "rating": 5},
        {"sentiment": "negative", "topic": "цены", "confidence": 0.5, "rating": 4},
        {"sentiment": "neutral", "topic": "другое", "confidence": 0.6, "rating": 3},
    ]
    s = cr.build_summary(items)
    assert s["sentiment_counts"]["positive"] == 1
    assert s["agreement_with_star_rating"]["matches"] == 2
    assert s["agreement_with_star_rating"]["accuracy"] == pytest.approx(0.667)


def test_end_to_end_csv(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("text,rating\nОтличное место,5\nУжасный сервис,1\n,3\nНормально,3\n", encoding="utf-8")
    out = tmp_path / "res.json"
    client = FakeClient([ok_payload([1, 2]), ok_payload([3], "neutral")])
    result = cr.main(["--source", "csv", "--csv", str(src), "--batch-size", "2", "--output", str(out)], client=client)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(result, ensure_ascii=False))
    assert saved["meta"]["n_reviews"] == 3  # пустой отзыв пропущен
    assert saved["meta"]["model"] == "fake/model:free"
    assert [i["text"] for i in saved["items"]] == ["Отличное место", "Ужасный сервис", "Нормально"]
    assert saved["items"][2]["sentiment"] == "neutral"


def test_hf_fetch_parses_rows(monkeypatch):
    rows = {"rows": [{"row": {"text": "Хорошо\\nочень", "rating": 5, "name_ru": "Кафе", "rubrics": "Кафе"}}]}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            return json.dumps(rows).encode()

    monkeypatch.setattr(cr.urllib.request, "urlopen", lambda url, timeout: Resp())
    assert cr.fetch_hf_reviews(1, 0) == [{"id": 1, "text": "Хорошо\nочень", "rating": 5, "name": "Кафе", "rubrics": "Кафе"}]


# --------------------------------------------------------------------------- настоящий HTTP-клиент


class FakeOpenRouter(BaseHTTPRequestHandler):
    """Имитирует OpenRouter: /models, /chat/completions, 429 с Retry-After и 400 на json_schema."""

    state: dict = {}

    def log_message(self, *a):
        pass

    def _send(self, code, payload, headers=None):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        assert self.path == "/api/v1/models"
        self._send(200, {"data": [
            {"id": "paid/model", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
            {"id": "some/free-model:free", "pricing": {"prompt": "0", "completion": "0"}, "supported_parameters": ["response_format"]},
        ]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.state.setdefault("bodies", []).append(body)
        self.state["auth"] = self.headers.get("Authorization")
        n = len(self.state["bodies"])
        if n == 1:
            return self._send(429, {"error": {"message": "Rate limit exceeded: per-minute"}}, {"Retry-After": "0"})
        if (body.get("response_format") or {}).get("type") == "json_schema":
            return self._send(400, {"error": {"message": "json_schema not supported by this provider"}})
        ids = [r["id"] for r in json.loads(body["messages"][1]["content"].split("<reviews>")[1].split("</reviews>")[0])]
        content = "```json\n" + json.dumps(ok_payload(ids), ensure_ascii=False) + "\n```"
        self._send(200, {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


@pytest.fixture
def server():
    FakeOpenRouter.state = {}
    httpd = HTTPServer(("127.0.0.1", 0), FakeOpenRouter)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/api/v1"
    httpd.shutdown()


def test_http_client_full_cycle(server, monkeypatch):
    monkeypatch.setattr(cr.time, "sleep", lambda s: None)
    client = cr.OpenAICompatibleClient("sk-or-test", server)
    client.model = client.pick_free_model()
    assert client.model == "some/free-model:free"  # платная модель пропущена

    usage = cr.Counter()
    out = cr.classify_batch(client, [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}], usage)
    assert [o["id"] for o in out] == [1, 2]
    bodies = FakeOpenRouter.state["bodies"]
    # 429 → повтор; 400 на json_schema → json_object; ответ в ```json``` корректно разобран
    assert [b["response_format"]["type"] for b in bodies] == ["json_schema", "json_schema", "json_object"]
    assert FakeOpenRouter.state["auth"] == "Bearer sk-or-test"
    assert usage["prompt_tokens"] == 10


def test_http_client_daily_limit_is_clear_error(server, monkeypatch):
    def handler(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self._send(429, {"error": {"message": "Rate limit exceeded: free-models-per-day"}})

    monkeypatch.setattr(FakeOpenRouter, "do_POST", handler)
    client = cr.OpenAICompatibleClient("k", server, "m:free")
    with pytest.raises(cr.LLMError, match="дневной лимит"):
        client.chat([{"role": "user", "content": "x"}], None)


# --------------------------------------------------------------------------- переключение моделей


class SwitchingClient(FakeClient):
    """Первая модель отвечает мусором, вторая — корректно."""

    fallback_models = ["good/model:free"]

    def __init__(self):
        super().__init__([])
        self.model = "bad/model:free"

    def chat(self, messages, response_format, max_tokens=4000):
        ids = [r["id"] for r in json.loads(messages[1]["content"].split("<reviews>")[1].split("</reviews>")[0])]
        out = "" if self.model == "bad/model:free" else json.dumps(ok_payload(ids), ensure_ascii=False)
        return out, {}


def test_main_switches_to_next_model_and_logs_raw(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("text\na\nb\nc\n", encoding="utf-8")
    out = tmp_path / "res.json"
    result = cr.main(["--source", "csv", "--csv", str(src), "--batch-size", "2", "--output", str(out)], client=SwitchingClient())
    assert result["meta"]["model"] == "good/model:free"
    assert all(i["model"] == "good/model:free" for i in result["items"])
    assert "bad/model:free" in (tmp_path / "debug_failed_responses.txt").read_text(encoding="utf-8")


def test_main_stops_when_all_models_fail(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("text\na\n", encoding="utf-8")
    client = SwitchingClient()
    client.fallback_models = []
    with pytest.raises(SystemExit, match="Ни одна модель"):
        cr.main(["--source", "csv", "--csv", str(src), "--output", str(tmp_path / "r.json")], client=client)


def test_fatal_error_does_not_switch_model(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("text\na\n", encoding="utf-8")
    client = FakeClient([cr.FatalLLMError("Исчерпан дневной лимит")])
    client.fallback_models = ["other:free"]
    with pytest.raises(SystemExit, match="дневной лимит"):
        cr.main(["--source", "csv", "--csv", str(src), "--output", str(tmp_path / "r.json")], client=client)


def test_candidates_order(monkeypatch):
    client = cr.OpenAICompatibleClient("k")
    monkeypatch.setattr(client, "list_free_models", lambda: [
        {"id": "poolside/laguna-xs-2.1:free", "context_length": 900000},
        {"id": "unknown/big:free", "context_length": 100000, "supported_parameters": ["response_format"]},
        {"id": "nvidia/nemotron-3-super-120b-a12b:free"},
    ])
    assert client.free_model_candidates() == [
        "nvidia/nemotron-3-super-120b-a12b:free", "unknown/big:free", "poolside/laguna-xs-2.1:free"
    ]


def test_reasoning_field_used_when_content_empty(server, monkeypatch):
    def handler(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self._send(200, {"choices": [{"message": {"content": "", "reasoning": '{"results": []}'}, "finish_reason": "stop"}]})

    monkeypatch.setattr(FakeOpenRouter, "do_POST", handler)
    text, _ = cr.OpenAICompatibleClient("k", server, "m:free").chat([{"role": "user", "content": "x"}], None)
    assert text == '{"results": []}'
