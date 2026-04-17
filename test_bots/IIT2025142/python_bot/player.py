from collections import Counter
import random

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from skeleton.states import STARTING_STACK

BIG_BLIND = 2
SMALL_BLIND = 1


class Player(Bot):

    def __init__(self):
        self.rng = random.Random()
        self.rank_value = {r: i for i, r in enumerate("23456789TJQKA", 2)}

        self.hands_played = 0
        self.decision_points = 0
        self.opp_bet_raise_events = 0
        self.opp_check_events = 0

        self.our_bet_attempts = 0
        self.opp_fold_to_bet = 0

        self.hand_opp_checks = 0
        self.we_bet_or_raised_this_hand = False

    def handle_new_round(self, game_state, round_state, active):
        self.hands_played += 1
        self.hand_opp_checks = 0
        self.we_bet_or_raised_this_hand = False

    def handle_round_over(self, game_state, terminal_state, active):
        previous = getattr(terminal_state, "previous_state", None)
        if previous is None:
            return

        if self.we_bet_or_raised_this_hand:
            opp_cards = []
            hands = getattr(previous, "hands", None)
            if hands and len(hands) > (1 - active):
                opp_cards = hands[1 - active] or []
            if len(opp_cards) < 2:
                self.opp_fold_to_bet += 1

    def _pot_size(self, round_state):
        stacks = getattr(round_state, "stacks", [STARTING_STACK, STARTING_STACK])
        if not stacks or len(stacks) < 2:
            return 0
        return max(0, 2 * STARTING_STACK - stacks[0] - stacks[1])

    def _safe_cards(self, card_list):
        out = []
        if not card_list:
            return out
        for c in card_list:
            if isinstance(c, str) and len(c) >= 2:
                r = c[0]
                s = c[1]
                if r in self.rank_value and s in "shdc":
                    out.append((r, s))
        return out

    def _is_connected_suited(self, c1, c2):
        v1 = self.rank_value[c1[0]]
        v2 = self.rank_value[c2[0]]
        gap = abs(v1 - v2)
        return c1[1] == c2[1] and 1 <= gap <= 3 and max(v1, v2) <= 11

    def _preflop_metrics(self, hole_cards):
        cards = self._safe_cards(hole_cards)
        if len(cards) < 2:
            return {
                "high": 2,
                "low": 2,
                "pair": False,
                "suited": False,
                "gap": 12,
                "ace": False,
                "broadway_count": 0,
            }
        r1, s1 = cards[0]
        r2, s2 = cards[1]
        v1 = self.rank_value[r1]
        v2 = self.rank_value[r2]
        high = max(v1, v2)
        low = min(v1, v2)
        return {
            "high": high,
            "low": low,
            "pair": r1 == r2,
            "suited": s1 == s2,
            "gap": abs(v1 - v2),
            "ace": (v1 == 14 or v2 == 14),
            "broadway_count": int(v1 >= 10) + int(v2 >= 10),
        }

    def _preflop_group(self, hole_cards):
        cards = self._safe_cards(hole_cards)
        if len(cards) < 2:
            return "trash"

        r1, s1 = cards[0]
        r2, s2 = cards[1]
        v1 = self.rank_value[r1]
        v2 = self.rank_value[r2]
        high = max(v1, v2)
        low = min(v1, v2)
        pair = r1 == r2
        suited = s1 == s2

        if pair and high >= 11:
            return "strong"
        if (high == 14 and low == 13) or (high == 14 and low == 12):
            return "strong"

        if pair and 7 <= high <= 10:
            return "medium"
        if (high == 14 and low == 11) or (high == 13 and low == 12):
            return "medium"

        if pair and high <= 6:
            return "weak_playable"
        if suited and abs(v1 - v2) <= 1 and high <= 11:
            return "weak_playable"
        if self._is_connected_suited(cards[0], cards[1]):
            return "weak_playable"
        if suited and high >= 11 and low >= 9:
            return "weak_playable"

        return "trash"

    def _straight_high(self, rank_values):
        vals = set(rank_values)
        if 14 in vals:
            vals.add(1)
        best = 0
        for start in range(1, 11):
            need = {start, start + 1, start + 2, start + 3, start + 4}
            if need.issubset(vals):
                best = start + 4
        return best

    def _made_hand_strength(self, hole_cards, board_cards):
        cards = self._safe_cards(hole_cards) + self._safe_cards(board_cards)
        if len(cards) < 5:
            return {
                "category": 0,
                "top_pair": False,
                "overpair": False,
                "flush_draw": False,
                "straight_draw": False,
            }

        ranks = [self.rank_value[r] for r, _ in cards]
        suits = [s for _, s in cards]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        counts = sorted(rank_counts.values(), reverse=True)
        is_quads = counts[0] == 4
        is_trips = counts[0] == 3
        pair_count = sum(1 for c in rank_counts.values() if c == 2)

        flush_suit = None
        for s, c in suit_counts.items():
            if c >= 5:
                flush_suit = s
                break

        straight_high = self._straight_high(ranks)
        is_straight = straight_high > 0

        is_straight_flush = False
        if flush_suit is not None:
            flush_ranks = [self.rank_value[r] for r, s in cards if s == flush_suit]
            if self._straight_high(flush_ranks) > 0:
                is_straight_flush = True

        if is_straight_flush:
            category = 8
        elif is_quads:
            category = 7
        elif is_trips and pair_count >= 1:
            category = 6
        elif flush_suit is not None:
            category = 5
        elif is_straight:
            category = 4
        elif is_trips:
            category = 3
        elif pair_count >= 2:
            category = 2
        elif pair_count == 1:
            category = 1
        else:
            category = 0

        board = self._safe_cards(board_cards)
        board_ranks = [self.rank_value[r] for r, _ in board]
        top_board = max(board_ranks) if board_ranks else 0

        hole = self._safe_cards(hole_cards)
        hvals = [self.rank_value[r] for r, _ in hole]

        top_pair = False
        overpair = False
        if category == 1 and top_board > 0 and len(hvals) == 2:
            if top_board in hvals:
                top_pair = True
            if hole[0][0] == hole[1][0] and hvals[0] > top_board:
                overpair = True

        flush_draw = False
        board_suits = [s for _, s in board]
        hole_suits = [s for _, s in hole]
        for suit in "shdc":
            if board_suits.count(suit) + hole_suits.count(suit) == 4:
                flush_draw = True
                break

        straight_draw = False
        if category < 4:
            uniq = set(ranks)
            if 14 in uniq:
                uniq.add(1)
            for start in range(1, 11):
                run = {start, start + 1, start + 2, start + 3, start + 4}
                if len(run.intersection(uniq)) >= 4:
                    straight_draw = True
                    break

        return {
            "category": category,
            "top_pair": top_pair,
            "overpair": overpair,
            "flush_draw": flush_draw,
            "straight_draw": straight_draw,
        }

    def _board_texture(self, board_cards):
        board = self._safe_cards(board_cards)
        if not board:
            return {"wet": False, "scary": False, "paired": False, "max_suit": 0}

        ranks = sorted({self.rank_value[r] for r, _ in board})
        suits = [s for _, s in board]
        max_suit = max(suits.count(s) for s in set(suits)) if suits else 0

        connected = False
        if len(ranks) >= 3:
            span = ranks[-1] - ranks[0]
            connected = span <= 4

        paired = len(ranks) < len(board)
        wet = (max_suit >= 3) or (connected and not paired)
        scary = (max_suit >= 3) or (connected and len(board) >= 4)
        return {"wet": wet, "scary": scary, "paired": paired, "max_suit": max_suit}

    def _opp_fold_to_bet_rate(self):
        if self.our_bet_attempts < 4:
            return 0.45
        return self.opp_fold_to_bet / max(1, self.our_bet_attempts)

    def _opp_raise_rate(self):
        if self.decision_points < 8:
            return 0.40
        return self.opp_bet_raise_events / max(1, self.decision_points)

    def _raise_to_fraction(self, round_state, active, fraction, pot_reference):
        try:
            min_raise, max_raise = round_state.raise_bounds()
        except Exception:
            return None

        pips = getattr(round_state, "pips", [0, 0])
        my_pip = pips[active] if pips and len(pips) > active else 0
        target = my_pip + max(BIG_BLIND, int(max(2, pot_reference) * fraction))
        amount = max(min_raise, min(target, max_raise))
        if amount < min_raise or amount > max_raise:
            return None
        return RaiseAction(amount)

    def _fallback_action(self, legal):
        if CheckAction in legal:
            return CheckAction()
        if CallAction in legal:
            return CallAction()
        return FoldAction()

    def _can_raise(self, legal):
        return RaiseAction in legal

    def _preflop_action(self, round_state, active, legal, hole_cards, hand_group, continue_cost, pot_size):
        opp_raise_rate = self._opp_raise_rate()
        in_position = (active == 0)
        m = self._preflop_metrics(hole_cards)
        defend_score = 0
        if m["pair"]:
            defend_score += 3
        if m["ace"]:
            defend_score += 2
        defend_score += m["broadway_count"]
        if m["suited"]:
            defend_score += 1
        if m["gap"] <= 2:
            defend_score += 1
        if m["high"] >= 11:
            defend_score += 1

        if hand_group == "strong":
            if self._can_raise(legal):
                action = self._raise_to_fraction(round_state, active, 0.65, max(10, pot_size + continue_cost))
                if action is not None:
                    self.we_bet_or_raised_this_hand = True
                    self.our_bet_attempts += 1
                    return action
            if CallAction in legal:
                return CallAction()
            return self._fallback_action(legal)

        if hand_group == "medium":
            if continue_cost <= 0:
                if self._can_raise(legal) and (in_position or self.rng.random() < 0.55):
                    action = self._raise_to_fraction(round_state, active, 0.50, max(8, pot_size + BIG_BLIND))
                    if action is not None:
                        self.we_bet_or_raised_this_hand = True
                        self.our_bet_attempts += 1
                        return action
                if CheckAction in legal:
                    return CheckAction()
                return self._fallback_action(legal)

            if continue_cost <= 8:
                if self._can_raise(legal) and in_position and continue_cost <= 4 and self.rng.random() < 0.30:
                    action = self._raise_to_fraction(round_state, active, 0.60, max(10, pot_size + continue_cost))
                    if action is not None:
                        self.we_bet_or_raised_this_hand = True
                        self.our_bet_attempts += 1
                        return action
                if CallAction in legal:
                    return CallAction()
            if CallAction in legal and continue_cost <= 12:
                return CallAction()
            if CallAction in legal and continue_cost <= 24 and defend_score >= 6 and opp_raise_rate < 0.70:
                return CallAction()
            if FoldAction in legal:
                return FoldAction()
            return self._fallback_action(legal)

        if hand_group == "weak_playable":
            call_cap = 6
            if opp_raise_rate > 0.60:
                call_cap = 4
            if continue_cost <= 0:
                if CheckAction in legal:
                    return CheckAction()
                if CallAction in legal:
                    return CallAction()
                return self._fallback_action(legal)
            if continue_cost <= call_cap and CallAction in legal:
                return CallAction()
            if CallAction in legal and continue_cost <= 9 and defend_score >= 3:
                return CallAction()
            if CallAction in legal and continue_cost <= 12 and defend_score >= 4 and opp_raise_rate < 0.65:
                return CallAction()
            if continue_cost <= BIG_BLIND and CallAction in legal:
                return CallAction()
            if FoldAction in legal:
                return FoldAction()
            return self._fallback_action(legal)

        if CallAction in legal and continue_cost <= 9 and defend_score >= 4 and opp_raise_rate < 0.62:
            return CallAction()
        if CallAction in legal and continue_cost <= 12 and defend_score >= 5 and opp_raise_rate < 0.58:
            return CallAction()
        if continue_cost <= BIG_BLIND and CallAction in legal:
            return CallAction()
        if continue_cost <= 0 and CheckAction in legal:
            if self._can_raise(legal) and in_position and self.rng.random() < 0.18:
                action = self._raise_to_fraction(round_state, active, 0.40, max(6, pot_size + BIG_BLIND))
                if action is not None:
                    self.we_bet_or_raised_this_hand = True
                    self.our_bet_attempts += 1
                    return action
            return CheckAction()
        if FoldAction in legal:
            return FoldAction()
        return self._fallback_action(legal)

    def _postflop_action(self, round_state, active, legal, continue_cost, pot_size, strength, texture, street, my_pip):
        category = strength["category"]
        top_pair = strength["top_pair"]
        overpair = strength["overpair"]
        draw = strength["flush_draw"] or strength["straight_draw"]

        fold_to_bet = self._opp_fold_to_bet_rate()
        raise_rate = self._opp_raise_rate()

        strong = (category >= 4) or (category >= 2 and not texture["paired"])
        medium = (category == 1 and (top_pair or overpair)) or (category == 3) or (draw and category <= 1)

        if continue_cost == 0:
            bluff_trigger = (self.hand_opp_checks >= 2) and texture["scary"] and (not strong) and (category <= 1)
            bluff_freq = 0.12
            if fold_to_bet > 0.55:
                bluff_freq = 0.18

            if strong and self._can_raise(legal):
                frac = 0.90 if category >= 5 else 0.65
                action = self._raise_to_fraction(round_state, active, frac, max(8, pot_size))
                if action is not None:
                    self.we_bet_or_raised_this_hand = True
                    self.our_bet_attempts += 1
                    return action

            if medium:
                if self._can_raise(legal) and (street == 5 or self.rng.random() < 0.44):
                    frac = 0.45 if street == 5 else 0.40
                    action = self._raise_to_fraction(round_state, active, frac, max(6, pot_size))
                    if action is not None:
                        self.we_bet_or_raised_this_hand = True
                        self.our_bet_attempts += 1
                        return action
                if CheckAction in legal:
                    return CheckAction()
                return self._fallback_action(legal)

            if self._can_raise(legal) and street in (3, 4):
                probe_freq = 0.16
                if raise_rate < 0.46:
                    probe_freq = 0.30
                if fold_to_bet > 0.55:
                    probe_freq = min(0.42, probe_freq + 0.10)
                if self.hand_opp_checks >= 1 and self.rng.random() < probe_freq:
                    action = self._raise_to_fraction(round_state, active, 0.35, max(6, pot_size))
                    if action is not None:
                        self.we_bet_or_raised_this_hand = True
                        self.our_bet_attempts += 1
                        return action

            if bluff_trigger and self._can_raise(legal) and self.rng.random() < bluff_freq:
                action = self._raise_to_fraction(round_state, active, 0.35, max(6, pot_size))
                if action is not None:
                    self.we_bet_or_raised_this_hand = True
                    self.our_bet_attempts += 1
                    return action

            if CheckAction in legal:
                return CheckAction()
            return self._fallback_action(legal)

        total = pot_size + continue_cost
        pot_odds = continue_cost / max(1, total)
        pressure = continue_cost / max(1, pot_size)
        facing_raise = my_pip > 0

        if facing_raise:
            if category <= 1:
                if FoldAction in legal:
                    return FoldAction()
                return self._fallback_action(legal)
            if category == 2 and pressure > 0.28:
                if FoldAction in legal:
                    return FoldAction()
                return self._fallback_action(legal)
            if category == 3 and pressure > 0.42 and street >= 4:
                if FoldAction in legal:
                    return FoldAction()
                return self._fallback_action(legal)
            if category == 4 and texture["max_suit"] >= 3 and pressure > 0.30:
                if FoldAction in legal:
                    return FoldAction()
                return self._fallback_action(legal)
            if category == 5 and texture["paired"] and pressure > 0.40:
                if FoldAction in legal:
                    return FoldAction()
                return self._fallback_action(legal)

        if strong:
            if self._can_raise(legal) and category >= 5:
                action = self._raise_to_fraction(round_state, active, 0.85, max(10, total))
                if action is not None:
                    self.we_bet_or_raised_this_hand = True
                    self.our_bet_attempts += 1
                    return action
            if self._can_raise(legal) and category >= 4 and self.rng.random() < 0.35:
                action = self._raise_to_fraction(round_state, active, 0.60, max(8, total))
                if action is not None:
                    self.we_bet_or_raised_this_hand = True
                    self.our_bet_attempts += 1
                    return action
            if CallAction in legal:
                return CallAction()
            return self._fallback_action(legal)

        if medium:
            call_threshold = 0.38
            if draw:
                call_threshold = 0.45
            if street >= 4:
                call_threshold -= 0.06
            if street >= 5:
                call_threshold -= 0.08
            if facing_raise:
                call_threshold -= 0.08
            if pressure > 0.60:
                call_threshold -= 0.06
            if raise_rate > 0.60:
                call_threshold += 0.03
            if pot_odds <= call_threshold and CallAction in legal:
                return CallAction()
            if FoldAction in legal:
                return FoldAction()
            return self._fallback_action(legal)

        bluff_catch_threshold = 0.18 if raise_rate > 0.65 else 0.11
        if street >= 5:
            bluff_catch_threshold -= 0.03
        if facing_raise:
            bluff_catch_threshold -= 0.03
        if pot_odds <= bluff_catch_threshold and CallAction in legal:
            return CallAction()
        if FoldAction in legal:
            return FoldAction()
        return self._fallback_action(legal)

    def get_action(self, game_state, round_state, active):
        try:
            legal = round_state.legal_actions()
        except Exception:
            return CheckAction()

        if not legal:
            return CheckAction()

        street = getattr(round_state, "street", 0)
        pips = getattr(round_state, "pips", [0, 0])
        my_pip = pips[active] if pips and len(pips) > active else 0
        opp_pip = pips[1 - active] if pips and len(pips) > (1 - active) else my_pip
        continue_cost = max(0, opp_pip - my_pip)
        pot_size = self._pot_size(round_state)

        self.decision_points += 1

        if continue_cost > 0:
            self.opp_bet_raise_events += 1
        elif street > 0 and getattr(round_state, "button", 0) > 1:
            self.hand_opp_checks += 1
            self.opp_check_events += 1

        hands = getattr(round_state, "hands", None)
        my_cards = []
        if hands and len(hands) > active:
            my_cards = hands[active] or []

        if street == 0:
            hand_group = self._preflop_group(my_cards)
            return self._preflop_action(round_state, active, legal, my_cards, hand_group, continue_cost, pot_size)

        board = getattr(round_state, "deck", []) or []
        strength = self._made_hand_strength(my_cards, board)
        texture = self._board_texture(board)

        return self._postflop_action(round_state, active, legal, continue_cost, pot_size, strength, texture, street, my_pip)


if __name__ == "__main__":
    run_bot(Player(), parse_args())
