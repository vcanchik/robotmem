"""python -m robotmem 入口

用法：
    python -m robotmem          # MCP Server（默认）
    python -m robotmem web      # Web UI
    python -m robotmem web --port 6889
"""

import sys


def main():
    args = sys.argv[1:]

    if args and args[0] == "web":
        # Web UI 模式
        from .config import load_config
        _cfg = load_config()
        port = _cfg.web_port
        host = "127.0.0.1"

        i = 1
        while i < len(args):
            if args[i] in ("--port", "-p") and i + 1 < len(args):
                try:
                    port = int(args[i + 1])
                    if not (1 <= port <= 65535):
                        raise ValueError
                except ValueError:
                    print(f"错误: 非法端口号 {args[i + 1]!r}（1-65535）")
                    sys.exit(1)
                i += 2
            elif args[i] in ("--host",) and i + 1 < len(args):
                host = args[i + 1]
                i += 2
            else:
                i += 1

        from .web import create_app
        app = create_app()
        print(f"robotmem Web UI: http://{host}:{port}")
        app.run(host=host, port=port, debug=False)
    else:
        # MCP Server 模式（默认）
        from .mcp_server import main as mcp_main
        mcp_main()


main()
