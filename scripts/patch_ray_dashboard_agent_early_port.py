#!/usr/bin/env python3
"""Patch Ray 2.55 dashboard agent startup for slow GPU nodes.

Raylet waits for ``dashboard_agent_listen_port_<node_id>`` before it finishes
startup. In Ray 2.55.1 the dashboard agent writes that file only after loading
all dashboard-agent modules and starting the HTTP server; on H20 nodes that can
take longer than raylet's internal wait window.

This idempotent patch makes the dashboard agent write the known fixed listen
port immediately after the grpc server starts, but only when
``SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT=1`` is present in the dashboard agent
environment. Dynamic listen port 0 is intentionally left untouched.
"""

from __future__ import annotations

from pathlib import Path

import ray.dashboard.agent as ray_dashboard_agent


MARKER = "SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT"
NEEDLE = """        modules = self._load_modules()
"""
INSERT = """        # SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT: Ray 2.55.1 writes this
        # after loading dashboard modules, which can exceed raylet's startup
        # wait on H20 nodes. For fixed worker ports, unblock raylet early and
        # let the normal write below refresh the same value after HTTP start.
        if (
            os.environ.get("SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT") == "1"
            and self.http_server
            and self.listen_port
        ):
            persist_port(
                self.session_dir,
                self.node_id,
                DASHBOARD_AGENT_LISTEN_PORT_NAME,
                self.listen_port,
            )

        modules = self._load_modules()
"""


def main() -> int:
    path = Path(ray_dashboard_agent.__file__).resolve()
    text = path.read_text()
    if MARKER in text:
        print(f"already patched: {path}")
        return 0
    if NEEDLE not in text:
        raise RuntimeError(f"patch anchor not found in {path}")
    path.write_text(text.replace(NEEDLE, INSERT, 1))
    print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
