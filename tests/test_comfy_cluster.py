from codesmith.comfy_cluster import (
    BackendConfig,
    BackendState,
    RouterState,
    _client_id_from_prompt_body,
    parse_backend,
)


def test_parse_backend_with_name() -> None:
    backend = parse_backend("pc1=http://192.168.1.10:8188")

    assert backend.name == "pc1"
    assert str(backend.url) == "http://192.168.1.10:8188/"


def test_router_selects_least_busy_backend() -> None:
    busy = BackendState(BackendConfig(name="busy", url="http://127.0.0.1:8188"), in_flight=2)
    idle = BackendState(BackendConfig(name="idle", url="http://127.0.0.1:8189"), in_flight=0)
    router = RouterState([busy, idle])

    import anyio

    selected = anyio.run(router.choose_backend)

    assert selected.config.name == "idle"


def test_router_uses_queue_load_for_selection() -> None:
    queued = BackendState(
        BackendConfig(name="queued", url="http://127.0.0.1:8188"),
        queue_pending=3,
    )
    free = BackendState(BackendConfig(name="free", url="http://127.0.0.1:8189"))
    router = RouterState([queued, free])

    import anyio

    selected = anyio.run(router.choose_backend)

    assert selected.config.name == "free"


def test_client_id_extracted_from_prompt_body() -> None:
    assert _client_id_from_prompt_body(b'{"client_id":"browser-1","prompt":{}}') == "browser-1"
    assert _client_id_from_prompt_body(b"not-json") is None
