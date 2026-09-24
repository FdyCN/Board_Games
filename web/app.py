"""
Web 人机对战后端（FastAPI，游戏无关）。

支持多个游戏（情书 / 璀璨宝石），通过 URL 路径区分：
    GET  /                    → 游戏选择页
    GET  /<game>/             → 该游戏的前端 HTML
    POST /<game>/api/new_game → 新开局
    GET  /<game>/api/state    → 当前状态（人类视角）
    POST /<game>/api/act      → 落子 + 自动跑完 AI 回合

启动：
    python web/app.py
    浏览器打开 http://127.0.0.1:8000
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from web.adapters import ADAPTERS

app = FastAPI(title="桌游人机对战")

STATIC = Path(__file__).resolve().parent / "static"

# 懒加载适配器（每个游戏加载一次模型）
_instances = {}


def get_adapter(game: str):
    if game not in ADAPTERS:
        raise HTTPException(status_code=404, detail=f"未知游戏: {game}")
    if game not in _instances:
        _instances[game] = ADAPTERS[game]()
    return _instances[game]


@app.get("/", response_class=HTMLResponse)
def picker():
    links = "".join(
        f'<li><a href="/{name}/">{name}</a></li>' for name in sorted(ADAPTERS)
    )
    return HTMLResponse(
        f"<h2>选择游戏</h2><ul>{links}</ul>"
    )


@app.get("/{game}/", response_class=HTMLResponse)
def game_page(game: str):
    get_adapter(game)  # 校验游戏存在
    html_path = STATIC / f"{game}.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"缺少前端页面: {game}.html")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/{game}/api/new_game")
def new_game(game: str):
    return get_adapter(game).new_game()


@app.get("/{game}/api/state")
def state(game: str):
    return get_adapter(game).view()


@app.post("/{game}/api/act")
def act(game: str, body: dict):
    try:
        return get_adapter(game).act(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print("桌游人机对战已启动: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
