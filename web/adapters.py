"""
Web 人机对战的「游戏适配层」。

每个游戏实现一个 GameAdapter，负责：
- 创建游戏 + 加载该游戏的最佳 AI 模型
- human_view(state)：把状态转成「人类视角」的 JSON（不泄露对手隐藏信息）
- parse_action(dict)：把前端的 JSON 转成游戏动作对象
- act(dict)：落子 + 自动跑完 AI 回合

当前支持：love_letter（3 人）、splendor（3 人）。
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch

from games.registry import create_game
from models.model_factory import create_model
from agents.neural_agent import NeuralAgent

HUMAN = 0
DEVICE = "cpu"


class GameAdapter:
    name = ""
    num_players = 3
    ckpt = ""

    def __init__(self):
        self.game = create_game(self.name, num_players=self.num_players)
        self.model = self._load_model()
        self.ai = NeuralAgent(self.model, device=DEVICE, name="AI") if self.model else None
        self.state = None

    def _load_model(self):
        return None

    # ---- 供子类实现 ----
    def human_view(self, state) -> dict:
        raise NotImplementedError

    def parse_action(self, d: dict):
        raise NotImplementedError

    # ---- 通用逻辑 ----
    def new_game(self) -> dict:
        self.state = self.game.reset()
        self.state = self._autoplay(self.state)
        return self.view()

    def view(self) -> dict:
        return {"game": self.name, **self.human_view(self.state)}

    def _autoplay(self, state):
        while not self.game.is_terminal(state) and self.game.get_current_player(state) != HUMAN:
            p = self.game.get_current_player(state)
            state, _, _, _ = self.game.step(self._ai_action(state, p))
        return state

    def _ai_action(self, state, player):
        obs = self.game.state_to_observation(state, player)
        legal = self.game.get_legal_actions(state)
        legal_idx = [self.game.action_to_index(a, state) for a in legal]
        idx, _ = self.ai.select_action(obs, legal_idx, deterministic=True)
        return self.game.index_to_action(idx, state)

    def act(self, d: dict) -> dict:
        if self.state is None or self.game.is_terminal(self.state):
            return self.new_game()
        if self.game.get_current_player(self.state) != HUMAN:
            return self.view()
        action = self.parse_action(d)
        if action not in self.game.get_legal_actions(self.state):
            raise ValueError(f"非法动作: {action}")
        self.state, _, _, _ = self.game.step(action)
        self.state = self._autoplay(self.state)
        return self.view()


# =====================================================================
# 情书
# =====================================================================
class LoveLetterAdapter(GameAdapter):
    name = "love_letter"
    num_players = 3
    ckpt = "data/love_letter/checkpoints/love_letter_3p_league_strong_v1/latest.pth"

    def _load_model(self):
        from games.love_letter.actions import PlayCardAction
        from games.love_letter.constants import CARD_NAMES

        self._PlayCardAction = PlayCardAction
        self._CARD_NAMES = CARD_NAMES
        model = create_model(
            obs_dim=self.game.observation_shape[0],
            action_size=self.game.action_space_size,
            encoder_type="gru",
            config="medium",
            aux_dim=self.game.auxiliary_shape[0] if self.game.auxiliary_shape else None,
            encoder_params=self.game.encoder_params,
        )
        ck = torch.load(self.ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        return model

    def human_view(self, state):
        me = HUMAN
        n = state.num_players
        players = []
        for p in range(n):
            players.append({
                "name": "你" if p == me else f"AI {p}",
                "tokens": state.tokens[p],
                "eliminated": state.eliminated[p],
                "protected": state.protected[p],
                "discards": list(state.discards[p]),
                "current": state.current_player == p,
            })
        legal = []
        if not state.game_over and state.current_player == me:
            legal = [
                {"card": a.card, "target": a.target, "guess": a.guess}
                for a in self.game.get_legal_actions(state)
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
            "known": [state.known[me][q] for q in range(n)],
            "legal_actions": legal,
            "events": [
                {"player": p, "card": c, "name": self._CARD_NAMES.get(c, str(c))}
                for p, c in state.events[-10:]
            ],
        }

    def parse_action(self, d):
        return self._PlayCardAction(card=d["card"], target=d.get("target", -1), guess=d.get("guess", -1))


# =====================================================================
# Splendor
# =====================================================================
class SplendorAdapter(GameAdapter):
    name = "splendor"
    num_players = 3
    ckpt = "data/splendor/checkpoints/mlp_medium_3p_v1/latest.pth"

    def _load_model(self):
        from games.splendor.actions import TakeGemsAction, ReserveCardAction, BuyCardAction, PassAction
        from games.splendor.constants import CardTier, GEM_NAMES

        self._TakeGemsAction = TakeGemsAction
        self._ReserveCardAction = ReserveCardAction
        self._BuyCardAction = BuyCardAction
        self._PassAction = PassAction
        self._CardTier = CardTier
        self._GEM_NAMES = GEM_NAMES

        model = create_model(
            obs_dim=self.game.observation_shape[0],
            action_size=self.game.action_space_size,
            encoder_type="mlp",
            config="medium",
        )
        ck = torch.load(self.ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        return model

    def human_view(self, state):
        from games.splendor.constants import CardTier

        me = HUMAN
        n = state.num_players

        def card_dict(c):
            return {
                "card_id": c.card_id, "tier": int(c.tier), "points": c.points,
                "bonus": int(c.bonus_color), "cost": list(c.cost),
            }

        open_cards = []
        for tier in (CardTier.TIER_1, CardTier.TIER_2, CardTier.TIER_3):
            for c in state.open_cards[tier]:
                open_cards.append(card_dict(c))

        nobles = [{"requirements": list(nob.requirements)} for nob in state.nobles]

        players = []
        for p in range(n):
            pl = state.players[p]
            players.append({
                "name": "你" if p == me else f"AI {p}",
                "gems": list(pl.gems),
                "bonuses": pl.get_total_bonuses(),
                "points": pl.get_score(),
                "reserved": [card_dict(c) for c in pl.reserved_cards],
                "nobles": len(pl.nobles),
                "current": state.current_player == p,
            })

        legal = []
        if not self.game.is_terminal(state) and self.game.get_current_player(state) == me:
            legal = [self._action_dict(a) for a in self.game.get_legal_actions(state)]

        return {
            "phase": "game_over" if self.game.is_terminal(state) else
                     ("human_turn" if self.game.get_current_player(state) == me else "ai_turn"),
            "winner": self.game.get_winner(state) if self.game.is_terminal(state) else -1,
            "me": me,
            "gem_bank": list(state.gem_bank),
            "open_cards": open_cards,
            "nobles": nobles,
            "players": players,
            "turn": state.turn_number,
            "winning_score": 15,
            "legal_actions": legal,
        }

    def _action_dict(self, a):
        if isinstance(a, self._TakeGemsAction):
            return {"type": "take_gems", "gems": list(a.gems)}
        if isinstance(a, self._ReserveCardAction):
            return {"type": "reserve", "tier": int(a.tier), "card_id": a.card_id}
        if isinstance(a, self._BuyCardAction):
            return {"type": "buy", "card_id": a.card_id, "from_reserved": a.from_reserved}
        if isinstance(a, self._PassAction):
            return {"type": "pass"}
        return {"type": "unknown"}

    def parse_action(self, d):
        t = d["type"]
        if t == "take_gems":
            return self._TakeGemsAction(gems=tuple(d["gems"]))
        if t == "reserve":
            return self._ReserveCardAction(tier=self._CardTier(int(d["tier"])), card_id=d.get("card_id"))
        if t == "buy":
            return self._BuyCardAction(card_id=int(d["card_id"]), from_reserved=bool(d.get("from_reserved", False)))
        if t == "pass":
            return self._PassAction()
        raise ValueError(f"未知动作类型: {t}")


# =====================================================================
ADAPTERS = {
    "love_letter": LoveLetterAdapter,
    "splendor": SplendorAdapter,
}

__all__ = ["GameAdapter", "LoveLetterAdapter", "SplendorAdapter", "ADAPTERS"]
