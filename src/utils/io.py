from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """将数据序列化为 JSON 文件，自动创建父目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(path: str | Path) -> Any:
    """从 JSON 文件读取数据。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
