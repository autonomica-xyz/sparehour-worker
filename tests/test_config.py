"""Regression tests for dworker.config: validation, coercion, endpoint
policy, and atomic private persistence. All offline, tmp-path based."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from dworker import config as config_mod
from dworker.config import ConfigError


def _config(**overrides):
    config = dict(config_mod.DEFAULTS)
    config.update(overrides)
    return config


# -- load / validate --------------------------------------------------------


def test_load_missing_file_returns_independent_defaults(tmp_path: Path):
    loaded = config_mod.load(tmp_path / "absent.json")
    assert loaded == config_mod.DEFAULTS
    loaded["capabilities"].append("x")
    loaded["labels"]["k"] = "v"
    assert config_mod.DEFAULTS["capabilities"] == []
    assert config_mod.DEFAULTS["labels"] == {}


def test_load_rejects_malformed_json(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot load"):
        config_mod.load(path)


def test_load_rejects_non_object_json(tmp_path: Path):
    path = tmp_path / "config.json"
    for stored in ("[1, 2]", '"text"', "42", "null"):
        path.write_text(stored, encoding="utf-8")
        with pytest.raises(ConfigError, match="JSON object"):
            config_mod.load(path)


def test_load_rejects_wrong_types_and_bad_ranges(tmp_path: Path):
    path = tmp_path / "config.json"
    for stored in (
        {"labels": ["not", "a", "dict"]},
        {"labels": {"tier": 3}},
        {"capabilities": "demo-gpu"},
        {"capabilities": ["ok", 7]},
        {"cpu_slots": -1},
        {"cpu_slots": True},
        {"idle_after_seconds": -5},
        {"dashboard_port": 70000},
        {"idle_only": "treu"},
        {"dashboard_port": 0},
        {"poll_seconds": -0.5},
        {"poll_seconds": "fast"},
        {"poll_seconds": float("nan")},
        {"token": 1234},
        {"dashboard_host": "0.0.0.0"},
    ):
        path.write_text(json.dumps(stored), encoding="utf-8")
        with pytest.raises(ConfigError, match="invalid config"):
            config_mod.load(path)


def test_validate_fills_missing_and_drops_unknown():
    result = config_mod.validate({"cpu_slots": 2, "mystery": [1]})
    assert result["cpu_slots"] == 2
    assert result["endpoint"] == ""
    assert "mystery" not in result
    result["capabilities"].append("x")
    assert config_mod.DEFAULTS["capabilities"] == []


def test_validate_heartbeat_must_be_below_lease():
    with pytest.raises(ConfigError, match="heartbeat_seconds"):
        config_mod.validate(_config(heartbeat_seconds=90, lease_seconds=90))
    with pytest.raises(ConfigError, match="heartbeat_seconds"):
        config_mod.validate(_config(heartbeat_seconds=120, lease_seconds=90))
    assert config_mod.validate(_config(heartbeat_seconds=15, lease_seconds=90))["lease_seconds"] == 90


def test_validate_endpoint_policy_enforced_on_load(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"endpoint": "http://control.example.com"}), encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid config"):
        config_mod.load(path)



def test_load_bad_token_value_is_not_echoed(tmp_path: Path):
    """A non-string token is rejected without leaking the secret into errors."""
    secret = "sekrit-token-value"
    path = tmp_path / "config.json"
    for bad in ({'nested': secret}, [secret], {'a': [secret]}):
        path.write_text(json.dumps({"token": bad}))
        with pytest.raises(ConfigError) as caught:
            config_mod.load(path)
        assert secret not in str(caught.value)


def test_model_catalog_requires_https_when_remote(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_catalog": "http://x.example.com/catalog.json"}))
    with pytest.raises(ConfigError):
        config_mod.load(path)
    path.write_text(json.dumps({"model_catalog": "https://x.example.com/catalog.json?k=v"}))
    assert config_mod.load(path)["model_catalog"].endswith("?k=v")
    path.write_text(json.dumps({"model_catalog": "/srv/catalog.lock"}))
    assert config_mod.load(path)["model_catalog"] == "/srv/catalog.lock"


def test_dashboard_host_coerces_localhost_rejects_public():
    assert config_mod.validate(_config(dashboard_host="localhost"))["dashboard_host"] == "127.0.0.1"
    assert config_mod.validate(_config(dashboard_host="::1"))["dashboard_host"] == "::1"
    assert config_mod.validate(_config(dashboard_host="::ffff:127.0.0.1"))["dashboard_host"] == "127.0.0.1"
    with pytest.raises(ConfigError):
        config_mod.validate(_config(dashboard_host="dashboard.internal"))
    with pytest.raises(ConfigError):
        config_mod.validate(_config(dashboard_host="10.0.0.5"))

    # The same policy backs dashboard.make_server's standalone guard.
    assert config_mod.validate_dashboard_host(" LOCALHOST ") == "127.0.0.1"
    assert config_mod.validate_dashboard_host("127.0.0.1") == "127.0.0.1"
    for host in ("0.0.0.0", "::", "localhost.evil.com", "localhost.", "0x7f000001"):
        with pytest.raises(ConfigError, match="loopback"):
            config_mod.validate_dashboard_host(host)


# -- coerce (CLI string -> typed value) -------------------------------------


def test_coerce_bool_is_strict():
    for word, expected in (("true", True), ("1", True), ("YES", True), ("on", True),
                           ("false", False), ("0", False), ("no", False), ("off", False)):
        assert config_mod.coerce("idle_only", word) is expected
    for typo in ("treu", "2", "", "enabled", "y", "nope"):
        with pytest.raises(ConfigError):
            config_mod.coerce("idle_only", typo)


def test_coerce_int_and_ranges():
    assert config_mod.coerce("cpu_slots", "4") == 4
    assert config_mod.coerce("cpu_slots", "0") == 0
    for bad in ("-1", "2.5", "abc", "true", "999999"):
        with pytest.raises(ConfigError):
            config_mod.coerce("cpu_slots", bad)


def test_coerce_float_and_ranges():
    assert config_mod.coerce("poll_seconds", "0.25") == 0.25
    for bad in ("0", "-1", "abc", "nan", "inf"):
        with pytest.raises(ConfigError):
            config_mod.coerce("poll_seconds", bad)


def test_coerce_lease_and_heartbeat_ranges():
    assert config_mod.coerce("heartbeat_seconds", "10") == 10
    for bad in ("0", "-3", "1.5", "soon"):
        with pytest.raises(ConfigError):
            config_mod.coerce("heartbeat_seconds", bad)
    with pytest.raises(ConfigError):
        config_mod.coerce("dashboard_port", "0")
    assert config_mod.coerce("dashboard_port", "7788") == 7788


def test_coerce_list_splits_commas():
    assert config_mod.coerce("capabilities", "a, b ,c") == ["a", "b", "c"]
    assert config_mod.coerce("capabilities", "") == []


def test_coerce_labels_parses_json_object():
    assert config_mod.coerce("labels", '{"tier":"trusted","zone":"a"}') == {
        "tier": "trusted",
        "zone": "a",
    }
    assert config_mod.coerce("labels", "{}") == {}
    for bad in ('{"a":1}', '["a","b"]', "{not json}", "x=1", '"s"'):
        with pytest.raises(ConfigError):
            config_mod.coerce("labels", bad)


def test_coerce_endpoint_validates_policy():
    assert config_mod.coerce("endpoint", "https://ctrl.example.com/") == "https://ctrl.example.com"
    with pytest.raises(ConfigError):
        config_mod.coerce("endpoint", "http://ctrl.example.com")


def test_coerce_unknown_key_rejected():
    with pytest.raises(ConfigError, match="unknown config key"):
        config_mod.coerce("nonexistent_key", "x")


# -- endpoint policy ---------------------------------------------------------


def test_validate_endpoint_accepts_https_and_loopback_http():
    assert config_mod.validate_endpoint("https://ctrl.example.com") == "https://ctrl.example.com"
    assert config_mod.validate_endpoint("HTTPS://Ctrl.Example.com:8443/") == \
        "https://ctrl.example.com:8443"
    assert config_mod.validate_endpoint("http://localhost:8080") == "http://localhost:8080"
    assert config_mod.validate_endpoint("http://127.0.0.1:8080/api") == "http://127.0.0.1:8080/api"
    assert config_mod.validate_endpoint("http://[::1]:8080") == "http://[::1]:8080"


def test_validate_endpoint_rejects_http_off_loopback():
    for endpoint in (
        "http://control.example.com",
        "http://192.168.1.10",
        "http://10.0.0.1:8080",
        "http://localhost.evil.com",
        "http://localhost.",          # trailing-dot alias is not literal localhost
        "http://127.1",               # not a literal IP address
        "http://2130706433",          # dword encoding of 127.0.0.1
        "http://0x7f000001",          # hex encoding of 127.0.0.1
        "http://[::ffff:808:808]",    # mapped 8.8.8.8, not loopback
    ):
        with pytest.raises(ConfigError):
            config_mod.validate_endpoint(endpoint)


def test_validate_endpoint_rejects_credentials_query_fragment():
    for endpoint in (
        "https://user:pw@ctrl.example.com",
        "https://user@ctrl.example.com",
        "https://ctrl.example.com/?token=abc",
        "https://ctrl.example.com#frag",
        "https://ctrl.example.com:99999",
        "ftp://ctrl.example.com",
        "ctrl.example.com",
        "https://",
        "",
        "  ",
        "https://evil.com\\@good.com",   # backslash makes userinfo, host good.com
        "https://bad host.example.com",
        "https://under_score.example.com",
    ):
        with pytest.raises(ConfigError):
            config_mod.validate_endpoint(endpoint)


# -- persistence --------------------------------------------------------------


def test_save_creates_private_file_from_the_start(tmp_path: Path, monkeypatch):
    """No write-then-chmod window: the swapped-in tempfile is already 0600."""
    path = tmp_path / "config.json"
    observed = {}
    real_replace = os.replace

    def spy_replace(src, dst):
        observed["mode"] = stat.S_IMODE(os.stat(src).st_mode)
        return real_replace(src, dst)

    monkeypatch.setattr(config_mod.os, "replace", spy_replace)
    with _umask(0o022):
        config_mod.save(_config(endpoint="https://x", token="sekrit"), path)
    assert observed["mode"] == 0o600
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_save_roundtrip_and_unknown_keys_not_persisted(tmp_path: Path):
    path = tmp_path / "config.json"
    config_mod.save(_config(token="t", cpu_slots=2, mystery="gone"), path)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert "mystery" not in stored
    loaded = config_mod.load(path)
    assert loaded["cpu_slots"] == 2
    assert loaded["token"] == "t"


def test_save_replaces_symlink_instead_of_writing_through(tmp_path: Path):
    target_dir = tmp_path / "elsewhere"
    target_dir.mkdir()
    protected = target_dir / "protected.json"
    protected.write_text('{"sentinel": true}', encoding="utf-8")
    path = tmp_path / "config.json"
    path.symlink_to(protected)

    config_mod.save(_config(endpoint="https://x", token="sekrit"), path)

    assert not path.is_symlink()
    assert json.loads(path.read_text())["token"] == "sekrit"
    assert protected.read_text() == '{"sentinel": true}'


def test_save_failure_preserves_previous_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "config.json"
    original = config_mod.save(_config(endpoint="https://x", token="first"), path)
    before = original.read_text(encoding="utf-8")

    def broken_replace(_src, _dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(config_mod.os, "replace", broken_replace)
    with pytest.raises(OSError):
        config_mod.save(_config(endpoint="https://x", token="second"), path)
    assert path.read_text(encoding="utf-8") == before
    # no tempfile is left behind in the config directory
    assert [entry.name for entry in tmp_path.iterdir()] == ["config.json"]


def test_save_rejects_invalid_config_without_touching_file(tmp_path: Path):
    path = tmp_path / "config.json"
    config_mod.save(_config(token="first"), path)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ConfigError):
        config_mod.save(_config(token="x", heartbeat_seconds=90, lease_seconds=90), path)
    assert path.read_text(encoding="utf-8") == before


def test_load_saved_config_contains_no_read_window_metadata(tmp_path: Path):
    path = tmp_path / "config.json"
    config_mod.save(_config(endpoint="https://x", token="sekrit"), path)
    assert path.stat().st_mode & 0o077 == 0


class _umask:
    def __init__(self, mask):
        self._mask = mask

    def __enter__(self):
        self._previous = os.umask(self._mask)

    def __exit__(self, *exc):
        os.umask(self._previous)
