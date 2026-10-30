"""Independent ITCH subset codec and as-of book. Not linked with the submission."""

from __future__ import annotations

import bisect
import heapq
import random
import struct


def _be(n: int, width: int) -> bytes:
    return n.to_bytes(width, "big")


def _stock(name: str) -> bytes:
    raw = name.encode("ascii")[:8]
    return raw + b" " * (8 - len(raw))


def msg_r(name: str) -> bytes:
    body = bytearray(39)
    body[0] = ord("R")
    body[11:19] = _stock(name)
    return bytes(body)


def msg_a(oid: int, side: str, shares: int, name: str, price: int) -> bytes:
    body = bytearray(36)
    body[0] = ord("A")
    body[11:19] = _be(oid, 8)
    body[19] = ord(side)
    body[20:24] = _be(shares, 4)
    body[24:32] = _stock(name)
    body[32:36] = _be(price, 4)
    return bytes(body)


def msg_e(side: str, shares: int, name: str) -> bytes:
    """An execution names a side and a symbol. The price bytes are unused."""
    body = bytearray(36)
    body[0] = ord("E")
    body[19] = ord(side)
    body[20:24] = _be(shares, 4)
    body[28:36] = _stock(name)
    return bytes(body)


class Book:
    """Live orders plus a min-id heap at each symbol, side, and price."""

    def __init__(self) -> None:
        self.orders: dict[int, dict] = {}
        self.heaps: dict[tuple[str, str, int], list[int]] = {}
        self.ladder: dict[tuple[str, str], list[int]] = {}
        self.live_n: dict[tuple[str, str, int], int] = {}

    def clone(self) -> "Book":
        other = Book()
        other.orders = {oid: dict(meta) for oid, meta in self.orders.items()}
        for oid, meta in other.orders.items():
            if meta["shares"] > 0:
                other._push(oid, meta)
        return other

    def _join(self, meta: dict) -> None:
        key = (meta["name"], meta["side"], meta["price"])
        count = self.live_n.get(key, 0)
        self.live_n[key] = count + 1
        if count:
            return
        arr = self.ladder.setdefault((meta["name"], meta["side"]), [])
        index = bisect.bisect_left(arr, meta["price"])
        if index == len(arr) or arr[index] != meta["price"]:
            arr.insert(index, meta["price"])

    def _leave(self, meta: dict) -> None:
        key = (meta["name"], meta["side"], meta["price"])
        count = self.live_n.get(key, 0) - 1
        if count > 0:
            self.live_n[key] = count
            return
        self.live_n.pop(key, None)
        arr = self.ladder.get((meta["name"], meta["side"]))
        if not arr:
            return
        index = bisect.bisect_left(arr, meta["price"])
        if index < len(arr) and arr[index] == meta["price"]:
            arr.pop(index)

    def _push(self, oid: int, meta: dict) -> None:
        heapq.heappush(self.heaps.setdefault((meta["name"], meta["side"], meta["price"]), []), oid)
        self._join(meta)

    def _drop(self, oid: int) -> None:
        meta = self.orders.pop(oid, None)
        if meta is not None:
            self._leave(meta)

    def _take_at(self, name: str, side: str, price: int, shares: int) -> int:
        heap = self.heaps.get((name, side, price))
        if heap is None:
            return shares
        while shares > 0 and heap:
            while heap:
                oid = heap[0]
                meta = self.orders.get(oid)
                if meta is None or meta["shares"] <= 0 or (meta["name"], meta["side"], meta["price"]) != (name, side, price):
                    heapq.heappop(heap)
                    continue
                break
            else:
                break
            oid = heap[0]
            meta = self.orders[oid]
            if shares >= meta["shares"]:
                shares -= meta["shares"]
                heapq.heappop(heap)
                self._drop(oid)
            else:
                meta["shares"] -= shares
                shares = 0
        return shares

    def _take(self, name: str, side: str, shares: int) -> None:
        while shares > 0:
            arr = self.ladder.get((name, side))
            if not arr:
                return
            price = arr[-1] if side == "B" else arr[0]
            left = self._take_at(name, side, price, shares)
            if left == shares:
                return
            shares = left

    def apply(self, body: bytes) -> None:
        kind = chr(body[0])
        if kind == "R":
            return
        if kind == "A":
            oid = int.from_bytes(body[11:19], "big")
            side = chr(body[19])
            shares = int.from_bytes(body[20:24], "big")
            name = body[24:32].decode("ascii").rstrip(" ")
            price = int.from_bytes(body[32:36], "big")
            self._drop(oid)
            meta = {"side": side, "shares": shares, "name": name, "price": price}
            self.orders[oid] = meta
            self._push(oid, meta)
            return
        if kind == "E":
            side = chr(body[19])
            shares = int.from_bytes(body[20:24], "big")
            name = body[28:36].decode("ascii").rstrip(" ")
            self._take(name, side, shares)
            return
        if kind == "X":
            oid = int.from_bytes(body[11:19], "big")
            shares = int.from_bytes(body[19:23], "big")
            cur = self.orders.get(oid)
            if cur is None:
                return
            if shares >= cur["shares"]:
                self._drop(oid)
            else:
                cur["shares"] -= shares
            return
        if kind == "D":
            self._drop(int.from_bytes(body[11:19], "big"))
            return
        if kind == "U":
            old = int.from_bytes(body[11:19], "big")
            new = int.from_bytes(body[19:27], "big")
            shares = int.from_bytes(body[27:31], "big")
            price = int.from_bytes(body[31:35], "big")
            cur = self.orders.get(old)
            if cur is None:
                return
            kept = dict(cur)
            self._drop(old)
            if new != old:
                self._drop(new)
            kept["shares"] = shares
            kept["price"] = price
            self.orders[new] = kept
            self._push(new, kept)


def msg_x(oid: int, shares: int) -> bytes:
    body = bytearray(23)
    body[0] = ord("X")
    body[11:19] = _be(oid, 8)
    body[19:23] = _be(shares, 4)
    return bytes(body)


def msg_d(oid: int) -> bytes:
    body = bytearray(19)
    body[0] = ord("D")
    body[11:19] = _be(oid, 8)
    return bytes(body)


def msg_u(old: int, new: int, shares: int, price: int) -> bytes:
    body = bytearray(35)
    body[0] = ord("U")
    body[11:19] = _be(old, 8)
    body[19:27] = _be(new, 8)
    body[27:31] = _be(shares, 4)
    body[31:35] = _be(price, 4)
    return bytes(body)


def frame(seq: int, body: bytes) -> bytes:
    return struct.pack(">IH", seq, len(body)) + body


def _line(orders: dict, symbol: str) -> str:
    bids: dict[int, tuple[int, int]] = {}
    asks: dict[int, tuple[int, int]] = {}
    for oid, meta in orders.items():
        if meta["name"] != symbol or meta["shares"] <= 0:
            continue
        bucket = bids if meta["side"] == "B" else asks
        shares, low = bucket.get(meta["price"], (0, oid))
        bucket[meta["price"]] = (shares + meta["shares"], min(low, oid))

    def fmt(bucket: dict[int, tuple[int, int]], reverse: bool) -> str:
        if not bucket:
            return "-"
        return ",".join(
            f"{price}:{shares}:{low}"
            for price, (shares, low) in sorted(bucket.items(), reverse=reverse)
        )

    return f"{symbol} B {fmt(bids, True)} A {fmt(asks, False)}"


def pinned_query_plan(events: dict[int, bytes], frames: list[bytes]) -> list[tuple[int, str]]:
    """Mid-span (seq, symbol) pairs after each rewrite — geometry only, no book fold.

    Each rewrite is followed by three queries between that rewrite and the tip at
    fractions from a fixed ladder; the second and third may name other symbols.
    Expected book lines are harvested from the verifier oracle engine in C++
    and cross-checked against the independent rebuild checker; they are not
    folded here, so rewrite counts can grow without a multi-minute Python fold.
    """
    if not frames:
        return []
    seqs = sorted(events)
    end_seq = seqs[-1]
    fracs = (1, 2, 3, 4, 5, 3, 2, 4, 1)  # over 6 → ~0.17 .. 0.83
    symbols = sorted(
        {
            events[s][24:32].decode("ascii").rstrip(" ")
            for s in seqs
            if events[s][:1] == b"A"
        }
    )
    out: list[tuple[int, str]] = []
    for index, blob in enumerate(frames):
        seq = int.from_bytes(blob[:4], "big")
        span = end_seq - seq
        touched = blob[6:][24:32].decode("ascii").rstrip(" ")
        for k in (0, 1, 2):
            num = fracs[(index * 3 + k) % len(fracs)]
            asked = seq + (span * num) // 6
            if asked >= end_seq and end_seq > seq:
                asked = end_seq - 1
            if asked < seq:
                asked = seq
            if k == 0:
                symbol = touched
            else:
                symbol = symbols[(index + num + k) % len(symbols)] if symbols else touched
            out.append((asked, symbol))
    return out


def pinned_queries(events: dict[int, bytes], frames: list[bytes]) -> list[tuple[int, str, str]]:
    """Plan plus a Python book fold. Prefer floor harvest in the grader for scale."""
    plan = pinned_query_plan(events, frames)
    if not plan:
        return []
    bodies = dict(events)
    seqs = sorted(bodies)
    ck_every = 256
    book = Book()
    checkpoints: list[tuple[int, Book]] = [(-1, Book())]
    for seq in seqs:
        book.apply(events[seq])
        if seq % ck_every == 0:
            checkpoints.append((seq, book.clone()))
    earliest: int | None = None
    out: list[tuple[int, str, str]] = []
    for index, blob in enumerate(frames):
        seq = int.from_bytes(blob[:4], "big")
        bodies[seq] = blob[6:]
        prev_earliest = earliest
        earliest = seq if earliest is None else min(earliest, seq)
        if prev_earliest is None or earliest < prev_earliest:
            book = Book()
            checkpoints = [(-1, Book())]
            for step in seqs:
                if step >= earliest:
                    break
                book.apply(bodies[step])
                if step % ck_every == 0:
                    checkpoints.append((step, book.clone()))
        pair = plan[3 * index : 3 * index + 3]
        snap_seq, snap = checkpoints[0]
        for checkpoint_seq, checkpoint in checkpoints:
            if checkpoint_seq < earliest:
                snap_seq, snap = checkpoint_seq, checkpoint
            else:
                break
        ordered = sorted(enumerate(pair), key=lambda row: (row[1][0], row[0]))
        cur = snap.clone()
        step_i = 0
        answers: dict[int, str] = {}
        for k, (asked, symbol) in ordered:
            while step_i < len(seqs) and seqs[step_i] <= snap_seq:
                step_i += 1
            while step_i < len(seqs) and seqs[step_i] <= asked:
                cur.apply(bodies[seqs[step_i]])
                step_i += 1
            answers[k] = _line(cur.orders, symbol)
            snap_seq = asked
        for k, (asked, symbol) in enumerate(pair):
            out.append((asked, symbol, answers[k]))
    return out


def answer_books(events: dict[int, bytes], specs: list[tuple[int, str | None]]) -> list[list[str]]:
    """One fold of the log, then a book at each requested sequence."""
    if not specs:
        return []
    order_ids = sorted(events)
    points: dict[int, list[int]] = {}
    for index, (seq, _only) in enumerate(specs):
        points.setdefault(seq, []).append(index)
    book = Book()
    cursor = 0
    out: list[list[str] | None] = [None] * len(specs)
    for seq in sorted(points):
        while cursor < len(order_ids) and order_ids[cursor] <= seq:
            book.apply(events[order_ids[cursor]])
            cursor += 1
        levels: dict[tuple[str, str, int], tuple[int, int]] = {}
        names = set()
        for oid, resting in book.orders.items():
            if resting["shares"] <= 0:
                continue
            names.add(resting["name"])
            key = (resting["name"], resting["side"], resting["price"])
            shares, low = levels.get(key, (0, oid))
            levels[key] = (shares + resting["shares"], min(low, oid))
        full = None
        if any(specs[index][1] is None for index in points[seq]):
            full = _format_book(levels, sorted(names), None)
        for index in points[seq]:
            only = specs[index][1]
            out[index] = full if only is None else _format_book(levels, [only], only)
    return out


def _format_book(levels: dict, wanted: list[str], only: str | None) -> list[str]:
    lines = []
    for name in wanted:
        bids = sorted(
            ((p, shares, low) for (n, side, p), (shares, low) in levels.items() if n == name and side == "B"),
            reverse=True,
        )
        asks = sorted(
            (p, shares, low) for (n, side, p), (shares, low) in levels.items() if n == name and side == "S"
        )
        if only is None and not bids and not asks:
            continue

        def fmt(rows: list[tuple[int, int, int]]) -> str:
            if not rows:
                return "-"
            return ",".join(f"{p}:{shares}:{low}" for p, shares, low in rows)

        lines.append(f"{name} B {fmt(bids)} A {fmt(asks)}")
    return lines


def _draw_price(rng: random.Random) -> int:
    """Most prices sit on a 100-tick ladder. The rest use a finer tick and miss that ladder."""
    if rng.random() < 0.7:
        return 10000 + rng.randrange(200) * 100
    tick = (1, 5, 10, 25)[rng.randrange(4)]
    return 10000 + rng.randrange(400) * tick


def generate(seed: int, spec: dict) -> dict:
    rng = random.Random(seed)
    symbols = [f"S{i:03d}" for i in range(spec["symbols"])]
    events: dict[int, bytes] = {}
    live: dict[int, dict] = {}
    ids: list[int] = []
    pos: dict[int, int] = {}
    by_name: dict[str, set[int]] = {}
    seq = 0
    blobs: list[bytes] = []
    current = bytearray()

    def emit(body: bytes) -> None:
        nonlocal seq, current
        seq += 1
        events[seq] = body
        current.extend(frame(seq, body))

    for name in symbols:
        emit(msg_r(name))

    heaps: dict[tuple[str, str, int], list[int]] = {}
    ladder: dict[tuple[str, str], list[int]] = {}
    live_n: dict[tuple[str, str, int], int] = {}

    def join(meta: dict) -> None:
        key = (meta["name"], meta["side"], meta["price"])
        count = live_n.get(key, 0)
        live_n[key] = count + 1
        if count:
            return
        arr = ladder.setdefault((meta["name"], meta["side"]), [])
        index = bisect.bisect_left(arr, meta["price"])
        if index == len(arr) or arr[index] != meta["price"]:
            arr.insert(index, meta["price"])

    def leave(meta: dict) -> None:
        key = (meta["name"], meta["side"], meta["price"])
        count = live_n.get(key, 0) - 1
        if count > 0:
            live_n[key] = count
            return
        live_n.pop(key, None)
        arr = ladder.get((meta["name"], meta["side"]))
        if not arr:
            return
        index = bisect.bisect_left(arr, meta["price"])
        if index < len(arr) and arr[index] == meta["price"]:
            arr.pop(index)

    def remember(oid: int, meta: dict) -> None:
        live[oid] = meta
        heapq.heappush(heaps.setdefault((meta["name"], meta["side"], meta["price"]), []), oid)
        join(meta)
        if oid not in pos:
            pos[oid] = len(ids)
            ids.append(oid)
        by_name.setdefault(meta["name"], set()).add(oid)

    def take_at(name: str, side: str, price: int, shares: int) -> int:
        heap = heaps.get((name, side, price), [])
        while shares > 0 and heap:
            while heap:
                top = heap[0]
                meta = live.get(top)
                if meta is None or meta["shares"] <= 0 or (meta["name"], meta["side"], meta["price"]) != (name, side, price):
                    heapq.heappop(heap)
                    continue
                break
            else:
                break
            top = heap[0]
            have = live[top]["shares"]
            if shares >= have:
                shares -= have
                heapq.heappop(heap)
                forget(top)
            else:
                live[top]["shares"] -= shares
                shares = 0
        return shares

    def sweep(name: str, side: str, shares: int) -> None:
        while shares > 0:
            arr = ladder.get((name, side))
            if not arr:
                return
            price = arr[-1] if side == "B" else arr[0]
            left = take_at(name, side, price, shares)
            if left == shares:
                return
            shares = left

    def forget(oid: int) -> None:
        meta = live.pop(oid)
        leave(meta)
        by_name[meta["name"]].discard(oid)
        i = pos.pop(oid)
        last = ids.pop()
        if last != oid:
            ids[i] = last
            pos[last] = i

    def pick() -> int:
        return ids[rng.randrange(len(ids))]

    def fresh_id() -> int:
        oid = rng.getrandbits(64)
        while oid in live or oid == 0:
            oid = rng.getrandbits(64)
        return oid

    def cut_shares(oid: int) -> int:
        resting = live[oid]["shares"]
        if rng.random() < 0.2:
            return resting + rng.randint(1, 40)
        return rng.randint(1, resting)

    def reduce(oid: int, shares: int) -> None:
        live[oid]["shares"] -= shares
        if live[oid]["shares"] <= 0:
            forget(oid)

    target = spec["messages"]
    while seq < target:
        roll = rng.random()
        if roll < 0.55 or not ids:
            if ids and rng.random() < 0.12:
                oid = pick()
                name = live[oid]["name"]
            else:
                oid = fresh_id()
                name = rng.choice(symbols)
            side = "B" if rng.random() < 0.5 else "S"
            shares = rng.randint(1, 400)
            price = _draw_price(rng)
            emit(msg_a(oid, side, shares, name, price))
            remember(oid, {"name": name, "side": side, "shares": shares, "price": price})
        elif roll < 0.75:
            oid = pick()
            meta = live[oid]
            shares = cut_shares(oid)
            emit(msg_e(meta["side"], shares, meta["name"]))
            sweep(meta["name"], meta["side"], shares)
        elif roll < 0.83:
            oid = pick()
            shares = cut_shares(oid)
            emit(msg_x(oid, shares))
            reduce(oid, shares)
        elif roll < 0.92:
            if rng.random() < 0.08:
                emit(msg_d(fresh_id()))
            else:
                oid = pick()
                emit(msg_d(oid))
                forget(oid)
        else:
            oid = pick()
            meta = live[oid]
            pool = by_name[meta["name"]]
            shares = rng.randint(1, 400)
            price = _draw_price(rng)
            if len(pool) > 1 and rng.random() < 0.35:
                new = rng.choice(tuple(pool))
                while new == oid:
                    new = rng.choice(tuple(pool))
                emit(msg_u(oid, new, shares, price))
                side = meta["side"]
                name = meta["name"]
                forget(oid)
                if new in live:
                    forget(new)
                remember(new, {"name": name, "side": side, "shares": shares, "price": price})
            else:
                new = fresh_id()
                emit(msg_u(oid, new, shares, price))
                forget(oid)
                remember(new, {"name": meta["name"], "side": meta["side"], "shares": shares, "price": price})
        if spec.get("chunk") and seq % spec["chunk"] == 0:
            blobs.append(bytes(current))
            current = bytearray()
    if current:
        blobs.append(bytes(current))

    rewrites = []
    adds = [s for s, body in events.items() if body[0] == ord("A") and s <= max(1, seq // 2)]
    late = [s for s, body in events.items() if body[0] == ord("A") and s > seq - 6000]
    if not adds:
        adds = [s for s, body in events.items() if body[0] == ord("A")]
    events_before = dict(events)
    for index in range(spec.get("backfill", 0)):
        pool = late if late and index % 5 == 0 else adds
        if not pool:
            break
        target_seq = rng.choice(pool)
        body = bytearray(events[target_seq])
        shares = rng.randint(1, 500)
        price = _draw_price(rng)
        body[20:24] = _be(shares, 4)
        body[32:36] = _be(price, 4)
        events[target_seq] = bytes(body)
        rewrites.append(frame(target_seq, bytes(body)))

    queries = []
    for _ in range(spec["queries"]):
        name = rng.choice(symbols)
        at = rng.randint(1, seq)
        queries.append((name, at))
    queries.append((symbols[0].lower(), rng.randint(1, seq)))
    queries.append(("NONE", rng.randint(1, seq)))
    snaps = [rng.randint(1, seq) for _ in range(spec["snaps"])]
    return {
        "events": events,
        "blobs": blobs,
        "events_before": events_before,
        "rewrite_frames": rewrites,
        "rewrites": b"".join(rewrites),
        "queries": queries,
        "snaps": snaps,
        "interleave": spec.get("interleave", False),
    }
