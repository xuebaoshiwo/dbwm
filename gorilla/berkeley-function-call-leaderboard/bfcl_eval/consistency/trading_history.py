"""Existential ordered history, without a size cutoff or guessed initial list."""

import json
from collections import Counter
from copy import deepcopy
from datetime import datetime

import z3


def key(record):
    record = deepcopy(record)
    if isinstance(record.get("amount"), (int, float)):
        record["amount"] = round(float(record["amount"]), 3)
    return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def timestamp(value):
    return (datetime.strptime(value, "%Y-%m-%d %H:%M:%S") - datetime(2024, 9, 1)).days * 86400 + (
        datetime.strptime(value, "%Y-%m-%d %H:%M:%S") - datetime(2024, 9, 1)).seconds


class History:
    def __init__(self, steps, solver):
        self.records, maxima = {}, Counter()
        fixed, modified = None, False
        for step in steps:
            name = step.get("tool", "").removeprefix("TradingBot.")
            obs = step.get("observation", {})
            if name in {"fund_account", "withdraw_funds"} and isinstance(obs, dict) and "error" not in obs:
                modified = True
            if step.get("tool", "").removeprefix("TradingBot.") != "get_transaction_history":
                continue
            obs = step.get("observation", {})
            if not isinstance(obs, dict) or "transaction_history" not in obs:
                continue
            if fixed is None and not modified and not any(step.get("args", {}).get(k) for k in ("start_date", "end_date")):
                fixed = deepcopy(obs["transaction_history"])
            counts = Counter()
            for record in obs["transaction_history"]:
                token = key(record)
                self.records[token] = deepcopy(record)
                self.records[token]["amount"] = round(float(record["amount"]), 3)
                counts[token] += 1
            maxima |= counts
        # An unobserved initial record can always be removed. Copies of an
        # identical record pass exactly the same date filters, so no more than
        # the largest observed multiplicity is needed.
        self.counts, self.ranks = {}, {}
        fixed_positions = {}
        for i, record in enumerate(fixed or []):
            fixed_positions.setdefault(key(record), []).append(i)
        for i, (token, maximum) in enumerate(maxima.items()):
            count = z3.IntVal(len(fixed_positions.get(token, []))) if fixed is not None else z3.Int(f"initial_history_count_{i}")
            solver.add(count >= 0, count <= maximum)
            if self.records[token]["type"] not in {"deposit", "withdrawal"} or self.records[token]["amount"] <= 0:
                solver.add(count == 0)
            self.counts[token] = count
            self.ranks[token] = [(z3.IntVal(fixed_positions[token][j]) if fixed is not None and j < len(fixed_positions.get(token, []))
                                 else z3.Int(f"initial_history_rank_{i}_{j}")) for j in range(maximum)]

    def observe(self, args, output, writes, require, index, equal):
        epoch = datetime(2024, 9, 1)
        start = datetime.strptime(args["start_date"], "%Y-%m-%d") if args.get("start_date") else datetime.min
        end = datetime.strptime(args["end_date"], "%Y-%m-%d") if args.get("end_date") else datetime.max
        low = (start - epoch).days * 86400 + (start - epoch).seconds
        high = (end - epoch).days * 86400 + (end - epoch).seconds
        included = {token for token, record in self.records.items() if low <= timestamp(record["timestamp"]) <= high}
        length = z3.simplify(z3.Sum([self.counts[token] for token in included])) if included else z3.IntVal(0)
        occurrences, positions = Counter(), {}
        previous_rank = None
        for j, record in enumerate(output):
            token = key(record)
            occurrence = occurrences[token]
            occurrences[token] += 1
            positions.setdefault(token, []).append(j)
            rank = self.ranks[token][occurrence]
            require(z3.Implies(j < length, self.counts[token] > occurrence), index)
            if previous_rank is not None:
                require(z3.Implies(j < length, previous_rank < rank), index)
            previous_rank = rank
        for token in self.records:
            prefix_count = z3.Sum([z3.If(j < length, 1, 0) for j in positions.get(token, [])])
            require(prefix_count == (self.counts[token] if token in included else 0), index)
        position = length
        for write in writes:
            moment = write["timestamp"]
            visible = z3.And(moment >= low, moment <= high)
            matches = []
            for j, record in enumerate(output):
                if set(record) == {"type", "amount", "timestamp"} and record["type"] == write["type"] and equal(record["amount"], write["amount"]):
                    matches.append(z3.And(position == j, moment == timestamp(record["timestamp"])))
            require(z3.Implies(visible, z3.Or(*matches)), index)
            position = position + z3.If(visible, 1, 0)
        require(position == len(output), index)

    def witness(self, model):
        records = []
        for token, count in self.counts.items():
            for rank in self.ranks[token][:model.eval(count, model_completion=True).as_long()]:
                records.append((model.eval(rank, model_completion=True).as_long(), self.records[token]))
        return [deepcopy(record) for _, record in sorted(records, key=lambda pair: pair[0])]
