"""Finite, observation-complete encodings of the mutable list state."""

import z3
import ast


def order_ids(steps):
    """All named identities, plus keys necessarily skipped by the counter."""
    ids, previous = set(), None
    for step in steps:
        name = step.get("tool", "").removeprefix("TradingBot.")
        args, obs = step.get("args", {}), step.get("observation", {})
        if name in {"activate_order", "cancel_order", "execute_order", "get_order_details"}:
            ids.add(args["order_id"])
        if not isinstance(obs, dict):
            continue
        if name == "get_order_history" and "history" in obs:
            ids.update(obs["history"])
        if name == "get_order_details" and isinstance(obs.get("error"), str) and "orders_id: " in obs["error"]:
            try:
                ids.update(ast.literal_eval(obs["error"].split("orders_id: ", 1)[1]))
            except (ValueError, SyntaxError, TypeError):
                pass
        if name == "place_order" and "order_id" in obs:
            current = obs["order_id"]
            ids.add(current)
            if previous is not None:
                ids.update(range(previous + 1, current))
            previous = current
    return sorted(ids)


class OrderKeys:
    def __init__(self, ids):
        self.initial = {i: z3.Bool(f"initial_order_exists_{i}") for i in ids}
        self.current = dict(self.initial)
        self.ranks = {i: z3.Int(f"initial_order_rank_{i}") for i in ids}
        self.created = []

    def contains(self, order_id, initial=False):
        return (self.initial if initial else self.current).get(order_id, z3.BoolVal(False))

    def observe(self, history):
        if len(history) != len(set(history)):
            return z3.BoolVal(False)
        if self.created and history[-len(self.created):] != self.created:
            return z3.BoolVal(False)
        initial = history[:-len(self.created)] if self.created else history
        conditions = [present == (i in initial) for i, present in self.initial.items()]
        conditions += [self.ranks[a] < self.ranks[b] for a, b in zip(initial, initial[1:])]
        return z3.And(*conditions)

    def place(self, order_id):
        conditions = [z3.Not(self.contains(order_id))]
        if self.created:
            previous = self.created[-1]
            conditions.append(order_id > previous)
            conditions.extend(self.contains(i) for i in range(previous + 1, order_id))
        self.created.append(order_id)
        self.current[order_id] = z3.BoolVal(True)
        return z3.And(*conditions)

    def witness(self, model):
        present = [i for i, expr in self.initial.items() if z3.is_true(model.eval(expr, model_completion=True))]
        return sorted(present, key=lambda i: model.eval(self.ranks[i], model_completion=True).as_long())


class Watchlist:
    def __init__(self, steps, symbols, solver, initial=None):
        self.symbols = symbols
        reads, removals, adds, present_checks = [], 0, 0, 0
        for step in steps:
            name = step.get("tool", "").removeprefix("TradingBot.")
            obs = step.get("observation", {})
            if not isinstance(obs, dict):
                continue
            if name == "get_watchlist" and "watchlist" in obs:
                reads.append(obs["watchlist"])
            if name == "remove_stock_from_watchlist" and "status" in obs:
                removals += 1
            if name == "add_to_watchlist" and "status" in obs:
                adds += obs["status"].endswith(" added to watchlist successfully.")
                present_checks += obs["status"].endswith(" is already in the watchlist.")
        # Every initial element either survives into a full read or has been
        # removed before it. Without reads, irrelevant elements can be erased;
        # retain at most one per successful removal or membership observation.
        bound = max(map(len, reads), default=present_checks) + removals
        if initial is None:
            for step in steps:
                name = step.get("tool", "").removeprefix("TradingBot.")
                obs = step.get("observation", {})
                if name == "get_watchlist" and isinstance(obs, dict) and "watchlist" in obs:
                    initial = obs["watchlist"]
                    break
                if name in {"add_to_watchlist", "remove_stock_from_watchlist"}:
                    break
        if initial is not None:
            bound = len(initial)
            self.initial_length = z3.IntVal(bound)
            self.initial_slots = [z3.IntVal(symbols.index(s)) for s in initial]
        else:
            self.initial_length = z3.Int("initial_watchlist_length")
            solver.add(self.initial_length >= 0, self.initial_length <= bound)
            self.initial_slots = [z3.Int(f"initial_watchlist_slot_{i}") for i in range(bound)]
            for slot in self.initial_slots:
                solver.add(slot >= 0, slot < len(symbols))
        self.length = self.initial_length
        self.slots = self.initial_slots + [z3.IntVal(0)] * adds

    def contains(self, symbol):
        code = self.symbols.index(symbol)
        return z3.Or(*[z3.And(i < self.length, slot == code) for i, slot in enumerate(self.slots)])

    def observe(self, values):
        conditions = [self.length == len(values)]
        if len(values) > len(self.slots):
            return z3.BoolVal(False)
        conditions.extend(self.slots[i] == self.symbols.index(s) for i, s in enumerate(values))
        self.length = z3.IntVal(len(values))
        self.slots = [z3.IntVal(self.symbols.index(s)) for s in values] + [z3.IntVal(0)] * (len(self.slots) - len(values))
        return z3.And(*conditions)

    def append(self, symbol):
        code = self.symbols.index(symbol)
        self.slots = [z3.simplify(z3.If(self.length == i, code, slot)) for i, slot in enumerate(self.slots)]
        self.length = z3.simplify(self.length + 1)

    def remove_first(self, symbol):
        code = self.symbols.index(symbol)
        position = z3.IntVal(-1)
        for i in reversed(range(len(self.slots))):
            position = z3.If(z3.And(i < self.length, self.slots[i] == code), i, position)
        self.slots = [z3.simplify(z3.If(i >= position, self.slots[i + 1] if i + 1 < len(self.slots) else 0, slot))
                      for i, slot in enumerate(self.slots)]
        self.length = z3.simplify(self.length - 1)

    def witness(self, model):
        count = model.eval(self.initial_length, model_completion=True).as_long()
        return [self.symbols[model.eval(slot, model_completion=True).as_long()]
                for slot in self.initial_slots[:count]]
