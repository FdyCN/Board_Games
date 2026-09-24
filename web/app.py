"""
情书（Love Letter）Web 人机对战后端（FastAPI）。

- 3 人对局：你（玩家 0）+ 2 个 AI（用训练好的最佳 GRU 模型）
- 后端持有完整游戏状态；只把「你可见的信息」暴露给前端（不含 AI 手牌）
- 你落子后，后端自动把两个 AI 的回合跑完，再回到你

启动：
    python web/app.py
    然后浏览器打开 http://127.0.0.1:8000
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from games.registry import create_game
from games.love_letter.actions import PlayCardAction
from games.love_letter.constants import CARD_NAMES
from models.model_factory import create_model
from agents.neural_agent import NeuralAgent

CKPT = "data/love_letter/checkpoints/love_letter_3p_league_strong_v1/latest.pth"
HUMAN = 0
DEVICE = "cpu"

app = FastAPI(title="情书人机对战")

# ===== 初始化游戏 + AI =====
game = create_game("love_letter", num_players=3)
model = create_model(
    obs_dim=game.observation_shape[0],
    action_size=game.action_space_size,
    encoder_type="gru",
    config="medium",
    aux_dim=game.auxiliary_shape[0] if game.auxiliary_shape else None,
    encoder_params=game.encoder_params,
)
ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ck["model_state_dict"])
model.eval()
ai_agent = NeuralAgent(model, device=DEVICE, name="AI")

_state = None


def ai_choose(state, player):
    """AI 选动作（确定性 argmax）。"""
    obs = game.state_to_observation(state, player)
    legal = game.get_legal_actions(state)
    legal_idx = [game.action_to_index(a, state) for a in legal]
    idx, _ = ai_agent.select_action(obs, legal_idx, deterministic=True)
    return game.index_to_action(idx, state)


def auto_play_ai(state):
    """把 AI 的回合跑完，直到轮到人类或游戏结束。"""
    while not state.game_over and state.current_player != HUMAN:
        action = ai_choose(state, state.current_player)
        state, _, _, _ = game.step(action)
    return state


def human_view(state):
    me = HUMAN
    n = state.num_players
    players = []
    for p in range(n):
        players.append({
            "name": "你" if p == me else f"AI {p}",
            "tokens": state.tokens[p],
            "eliminated": state.eliminated[p],
            "protected": state.protected[p],
            "discards": state.discards[p],
            "current": state.current_player == p,
        })

    legal = []
    if not state.game_over and state.current_player == me:
        legal = [
            {"card": a.card, "target": a.target, "guess": a.guess}
            for a in game.get_legal_actions(state)
        ]

    return {
        "phase": "game_over" if state.game_over else
                 ("human_turn" if state.current_player == me else "ai_turn"),
        "winner": state.winner,
        "me": me,
        "my_hand": list(state.hands[me]),
        "players": players,
        "deck_size": len(state.deck),
        "round": state.round_number,
        "turn": state.turn_number,
        "target_tokens": state.target_tokens,
        # 你私底下知道每个对手是哪张牌（None=未知）；绝不暴露 AI 真实手牌
        "known": [state.known[me][q] for q in range(n)],
        "legal_actions": legal,
        "events": [
            {"player": p, "card": c, "name": CARD_NAMES.get(c, str(c))}
            for p, c in state.events[-10:]
        ],
    }


class ActionBody(BaseModel):
    card: int
    target: int = -1
    guess: int = -1


@app.post("/api/new_game")
def new_game():
    global _state
    _state = game.reset()
    _state = auto_play_ai(_state)
    return human_view(_state)


@app.get("/api/state")
def get_state():
    global _state
    if _state is None:
        return new_game()
    return human_view(_state)


@app.post("/api/act")
def act(body: ActionBody):
    global _state
    if _state is None or _state.game_over:
        return new_game()
    if _state.current_player != HUMAN:
        return human_view(_state)

    action = PlayCardAction(card=body.card, target=body.target, guess=body.guess)
    if action not in game.get_legal_actions(_state):
        return JSONResponse(status_code=400, content={"error": "非法动作"})

    _state, _, _, _ = game.step(action)
    _state = auto_play_ai(_state)
    return human_view(_state)


@app.get("/", response_class=HTMLResponse)
def index():
    html = Path(__file__).with_name("index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


if __name__ == "__main__":
    import uvicorn

    print("情书人机对战已启动: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
