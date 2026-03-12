"""__main__.py 单元测试 — CLI 参数解析逻辑

注意: robotmem/__main__.py 在模块级别调用 main()，
不能直接导入，所以这里只测试参数解析逻辑本身。
"""

import pytest


def _parse_web_args(args: list[str]) -> tuple[int, str]:
    """复制 __main__.py 的参数解析逻辑用于测试"""
    port = 7878
    host = "127.0.0.1"

    i = 1
    while i < len(args):
        if args[i] in ("--port", "-p") and i + 1 < len(args):
            try:
                port = int(args[i + 1])
                if not (1 <= port <= 65535):
                    raise ValueError
            except ValueError:
                raise ValueError(f"非法端口号 {args[i + 1]!r}")
            i += 2
        elif args[i] in ("--host",) and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        else:
            i += 1

    return port, host


class TestWebArgsParsing:
    """web 模式参数解析"""

    def test_default_port_and_host(self):
        port, host = _parse_web_args(["web"])
        assert port == 7878
        assert host == "127.0.0.1"

    def test_custom_port(self):
        port, host = _parse_web_args(["web", "--port", "7888"])
        assert port == 7888

    def test_short_port(self):
        port, host = _parse_web_args(["web", "-p", "8080"])
        assert port == 8080

    def test_custom_host(self):
        port, host = _parse_web_args(["web", "--host", "0.0.0.0"])
        assert host == "0.0.0.0"

    def test_port_and_host(self):
        port, host = _parse_web_args(["web", "--port", "9999", "--host", "0.0.0.0"])
        assert port == 9999
        assert host == "0.0.0.0"

    def test_invalid_port_str(self):
        with pytest.raises(ValueError):
            _parse_web_args(["web", "--port", "abc"])

    def test_port_zero(self):
        with pytest.raises(ValueError):
            _parse_web_args(["web", "--port", "0"])

    def test_port_too_large(self):
        with pytest.raises(ValueError):
            _parse_web_args(["web", "--port", "99999"])

    def test_port_negative(self):
        with pytest.raises(ValueError):
            _parse_web_args(["web", "--port", "-1"])

    def test_unknown_args_skipped(self):
        port, host = _parse_web_args(["web", "--unknown", "value", "--port", "7888"])
        assert port == 7888

    def test_port_at_end_without_value(self):
        """--port 在末尾没有值 → 跳过"""
        port, host = _parse_web_args(["web", "--port"])
        assert port == 7878  # 没有值，保持默认

    def test_host_at_end_without_value(self):
        """--host 在末尾没有值 → 跳过"""
        port, host = _parse_web_args(["web", "--host"])
        assert host == "127.0.0.1"


class TestModeDetection:
    """模式检测"""

    def test_no_args_is_mcp(self):
        args = []
        is_web = bool(args) and args[0] == "web"
        assert is_web is False

    def test_web_arg_is_web(self):
        args = ["web"]
        is_web = bool(args) and args[0] == "web"
        assert is_web is True

    def test_other_args_is_mcp(self):
        args = ["something"]
        is_web = bool(args) and args[0] == "web"
        assert is_web is False

    def test_web_with_options(self):
        args = ["web", "--port", "8080"]
        is_web = bool(args) and args[0] == "web"
        assert is_web is True
