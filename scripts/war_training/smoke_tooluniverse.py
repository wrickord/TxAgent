#!/usr/bin/env python3
"""Confirm tooluniverse is importable and a light FDA tool runs."""

from __future__ import annotations

import json
import sys


def main() -> int:
    import tooluniverse
    from tooluniverse import ToolUniverse

    print(f"tooluniverse {getattr(tooluniverse, '__version__', 'unknown')}")
    print(f"file {tooluniverse.__file__}")

    tu = ToolUniverse()
    tu.load_tools(
        categories=["openfda_labels"],
        include_tools=["FDA_search_drug_labels"],
    )
    result = tu.run(
        {
            "name": "FDA_search_drug_labels",
            "arguments": {
                "drug_name": "minoxidil",
                "limit": 1,
            },
        }
    )
    if isinstance(result, dict):
        keys = list(result.keys())[:12]
        print(f"FDA_search_drug_labels keys={keys}")
        err = result.get("error") or result.get("error_message")
        if err:
            print(f"tool error: {err}")
            print(json.dumps(result, indent=2, default=str)[:2000])
            return 1
        print("FDA smoke test ok")
        return 0
    print(f"unexpected result type {type(result)}: {str(result)[:500]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
