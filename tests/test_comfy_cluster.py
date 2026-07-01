from codesmith.comfy_cluster import BackendConfig, BackendState, RouterState, parse_backend


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
