// Reference: per-symbol checkpoints plus order-diff propagation after rewrites.
//
// bookd: as-of ITCH books with per-symbol checkpointed replay.
//
// Every event is routed statically to one symbol (A/E by stock; X/D/U by the
// symbol of their order id's last add, which is how the reference attributes
// them when walking history) and stored as a compact record in that symbol's
// seq-ordered list.  Each symbol keeps book checkpoints every kCk records; an
// as-of query loads the nearest checkpoint and replays at most kCk records.
// Books index price levels by stable per-symbol price ids with rank bitmaps,
// so every book operation is O(1).
//
// A rewritten add does not invalidate the later checkpoints: the difference
// between the old and new timelines (usually just the rewritten order) is
// pushed forward through the records and attached to each affected checkpoint
// as a patch.  Executions record how deep they reached so most records can be
// skipped; when an execution could touch a differing order it is re-simulated
// exactly in both timelines.  All of this happens synchronously inside APPLY,
// before READY is sent; there is no idle-time or deferred work of any kind.
//
// When a trace does something static routing cannot reproduce exactly (an id
// shared across symbols together with replaces or tip reads, unsorted or
// duplicate appends, events at seq 0) the engine replays every APPLY so far
// into the reference implementation and stays there.
//
// QUERY always computes its answer directly from the book; APPLY always fully
// completes before READY is sent.
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <map>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// ---------------------------------------------------------------- output
struct Out {
    char* p = nullptr;
    size_t n = 0, cap = 0;
    void reserve(size_t extra) {
        if (n + extra <= cap) return;
        size_t nc = cap ? cap : 1 << 16;
        while (nc < n + extra) nc *= 2;
        p = static_cast<char*>(realloc(p, nc));
        if (!p) abort();
        cap = nc;
    }
    void put(const char* s, size_t k) {
        reserve(k);
        memcpy(p + n, s, k);
        n += k;
    }
    void put(const std::string& s) { put(s.data(), s.size()); }
    void flush() {
        size_t off = 0;
        while (off < n) {
            ssize_t w = ::write(STDOUT_FILENO, p + off, n - off);
            if (w <= 0) {
                if (w < 0 && errno == EINTR) continue;
                _exit(2);
            }
            off += static_cast<size_t>(w);
        }
        n = 0;
    }
};
Out out;

// Exit paths flush what the reference would already have written.
[[noreturn]] void die(int code) {
    out.flush();
    _exit(code);
}
[[noreturn]] void die_abort() {
    out.flush();
    abort();
}

// ---------------------------------------------------------------- input
struct In {
    uint8_t* buf = nullptr;
    size_t cap = 0, pos = 0, len = 0;
    void init() {
        cap = 1 << 22;
        buf = static_cast<uint8_t*>(malloc(cap));
    }
    // Make room for at least `need` more bytes after len.
    void room(size_t need) {
        if (pos > 0 && (len - pos + need > cap - len || pos > cap / 2)) {
            memmove(buf, buf + pos, len - pos);
            len -= pos;
            pos = 0;
        }
        if (len + need > cap) {
            size_t nc = cap;
            while (len + need > nc) nc *= 2;
            uint8_t* nb = static_cast<uint8_t*>(realloc(buf, nc));
            if (!nb) die_abort();
            buf = nb;
            cap = nc;
        }
    }
    // Blocking read of more data; flushes pending output first.
    void more();
    // Returns pointer to a line (without '\n') and its length; advances.
    std::pair<const char*, size_t> line() {
        size_t scan = pos;
        while (true) {
            void* nl = memchr(buf + scan, '\n', len - scan);
            if (nl) {
                size_t e = static_cast<uint8_t*>(nl) - buf;
                const char* s = reinterpret_cast<const char*>(buf + pos);
                size_t k = e - pos;
                pos = e + 1;
                return {s, k};
            }
            scan = len - pos;
            more();
            scan += pos;
        }
    }
    // Ensure n bytes are available contiguously at pos.
    const uint8_t* bytes(size_t n) {
        if (len - pos < n) {
            room(n - (len - pos));
            while (len - pos < n) more();
        }
        const uint8_t* r = buf + pos;
        pos += n;
        return r;
    }
};
In in;

void In::more() {
    out.flush();
    room(1 << 16);
    while (true) {
        ssize_t r = ::read(STDIN_FILENO, buf + len, cap - len);
        if (r > 0) {
            len += static_cast<size_t>(r);
            return;
        }
        if (r < 0 && errno == EINTR) continue;
        _exit(2);
    }
}

// std::stoull-like parse: whitespace, optional sign, decimal digits.
// Throws (aborts) like the std:: functions on malformed input.
uint64_t parse_u64(const char* s, size_t n, bool* neg_out = nullptr) {
    size_t i = 0;
    while (i < n && (s[i] == ' ' || (s[i] >= '\t' && s[i] <= '\r'))) ++i;
    bool neg = false;
    if (i < n && (s[i] == '+' || s[i] == '-')) {
        neg = s[i] == '-';
        ++i;
    }
    if (i >= n || s[i] < '0' || s[i] > '9') die_abort();
    uint64_t v = 0;
    bool ovf = false;
    while (i < n && s[i] >= '0' && s[i] <= '9') {
        uint64_t d = static_cast<uint64_t>(s[i] - '0');
        if (v > (UINT64_MAX - d) / 10) ovf = true;
        v = v * 10 + d;
        ++i;
    }
    if (ovf) die_abort();
    if (neg_out) *neg_out = neg;
    return neg ? (0 - v) : v;
}

int parse_int(const char* s, size_t n) {
    // std::stoi: strtol then range check against int.
    size_t i = 0;
    while (i < n && (s[i] == ' ' || (s[i] >= '\t' && s[i] <= '\r'))) ++i;
    bool neg = false;
    if (i < n && (s[i] == '+' || s[i] == '-')) {
        neg = s[i] == '-';
        ++i;
    }
    if (i >= n || s[i] < '0' || s[i] > '9') die_abort();
    long long v = 0;
    while (i < n && s[i] >= '0' && s[i] <= '9') {
        v = v * 10 + (s[i] - '0');
        if (v > (1LL << 40)) die_abort();
        ++i;
    }
    if (neg) v = -v;
    if (v > INT32_MAX || v < INT32_MIN) die_abort();
    return static_cast<int>(v);
}

// ---------------------------------------------------------------- decoding
inline uint32_t be32(const uint8_t* p) {
    return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
}
inline uint64_t be64(const uint8_t* p) { return (uint64_t)be32(p) << 32 | be32(p + 4); }

struct Ev {
    uint64_t id = 0, id2 = 0, stock = 0;
    uint32_t seq = 0, shares = 0, price = 0;
    uint32_t d = 0xffffffffu, d2 = 0xffffffffu;      // dense ids
    uint32_t nref[2] = {0xffffffffu, 0xffffffffu};   // next in id ref list: (evidx << 1 | field)
    int32_t rsym = -1;                               // symbol holding this event's record
    char type = 0, side = 0;
};

// Returns false for frames the reference ignores entirely.
bool decode(uint32_t seq, const uint8_t* b, uint16_t len, Ev& e) {
    e = Ev();
    e.seq = seq;
    if (len < 1) return true;  // zero-length frame: a null event that still occupies seq
    e.type = static_cast<char>(b[0]);
    switch (e.type) {
        case 'R':
            if (len < 19) return false;
            memcpy(&e.stock, b + 11, 8);
            return true;
        case 'A':
            if (len < 36) return false;
            e.id = be64(b + 11);
            e.side = static_cast<char>(b[19]);
            e.shares = be32(b + 20);
            memcpy(&e.stock, b + 24, 8);
            e.price = be32(b + 32);
            return true;
        case 'E':
            if (len < 36) return false;
            e.side = static_cast<char>(b[19]);
            e.shares = be32(b + 20);
            e.price = be32(b + 24);
            memcpy(&e.stock, b + 28, 8);
            return true;
        case 'X':
            if (len < 23) return false;
            e.id = be64(b + 11);
            e.shares = be32(b + 19);
            return true;
        case 'D':
            if (len < 19) return false;
            e.id = be64(b + 11);
            return true;
        case 'U':
            if (len < 35) return false;
            e.id = be64(b + 11);
            e.id2 = be64(b + 19);
            e.shares = be32(b + 27);
            e.price = be32(b + 31);
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------- number formatting
const char kDig2[] =
    "00010203040506070809101112131415161718192021222324252627282930313233343536373839"
    "40414243444546474849505152535455565758596061626364656667686970717273747576777879"
    "8081828384858687888990919293949596979899";

inline char* put_u32(char* p, uint32_t v) {
    uint32_t n = v < 100000 ? (v < 100 ? (v < 10 ? 1 : 2) : (v < 1000 ? 3 : (v < 10000 ? 4 : 5)))
                            : (v < 10000000 ? (v < 1000000 ? 6 : 7) : (v < 100000000 ? 8 : (v < 1000000000 ? 9 : 10)));
    char* e = p + n;
    char* q = e;
    while (v >= 100) {
        uint32_t t = v / 100;
        uint32_t r = v - t * 100;
        q -= 2;
        memcpy(q, kDig2 + 2 * r, 2);
        v = t;
    }
    if (v >= 10)
        memcpy(q - 2, kDig2 + 2 * v, 2);
    else
        q[-1] = static_cast<char>('0' + v);
    return e;
}

inline char* put_u64(char* p, uint64_t v) {
    if (v <= 0xffffffffu) return put_u32(p, static_cast<uint32_t>(v));
    char tmp[24];
    char* e = tmp + 24;
    char* s = e;
    while (v >= 100) {
        uint64_t q = v / 100;
        uint32_t r = static_cast<uint32_t>(v - q * 100);
        s -= 2;
        memcpy(s, kDig2 + 2 * r, 2);
        v = q;
    }
    if (v >= 10) {
        s -= 2;
        memcpy(s, kDig2 + 2 * v, 2);
    } else {
        *--s = static_cast<char>('0' + v);
    }
    size_t k = static_cast<size_t>(e - s);
    memcpy(p, s, k);
    return p + k;
}

// ---------------------------------------------------------------- hash map
// u64 -> u32 map: open addressing, linear probing, Fibonacci hashing.
struct U64Map {
    struct E {
        uint64_t k;
        uint32_t v, pad;
    };
    std::vector<E> t;
    size_t mask = 0, n = 0;
    int shift = 64;
    static constexpr uint32_t kNone = 0xffffffffu;
    void init(size_t cap) {
        size_t c = 16;
        int lg = 4;
        while (c < cap * 2) {
            c *= 2;
            ++lg;
        }
        t.assign(c, E{0, kNone, 0});
        mask = c - 1;
        shift = 64 - lg;
        n = 0;
    }
    size_t home(uint64_t k) const { return static_cast<size_t>((k * 0x9E3779B97F4A7C15ull) >> shift); }
    uint32_t find(uint64_t k) const {
        if (!mask) return kNone;
        for (size_t i = home(k);; i = (i + 1) & mask) {
            const E& e = t[i];
            if (e.v == kNone) return kNone;
            if (e.k == k) return e.v;
        }
    }
    // Returns the existing value, or inserts v and returns v.
    uint32_t insert(uint64_t k, uint32_t v) {
        if (!mask || (n + 1) * 2 > mask + 1) grow();
        for (size_t i = home(k);; i = (i + 1) & mask) {
            E& e = t[i];
            if (e.v == kNone) {
                e.k = k;
                e.v = v;
                ++n;
                return v;
            }
            if (e.k == k) return e.v;
        }
    }
    void grow() {
        std::vector<E> old;
        old.swap(t);
        init(std::max<size_t>(16, (n + 1) * 2));
        for (const E& e : old)
            if (e.v != kNone) insert(e.k, e.v);
    }
};

constexpr uint32_t kNil = 0xffffffffu;

namespace ref {
// Reference engine (verbatim starter logic; output to a string).
constexpr int kSnapEvery = 14000;

uint32_t be32(const uint8_t* p) {
    return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
}
uint64_t be64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
    return v;
}

std::string trim_stock(const char* stock) {
    int n = 8;
    while (n > 0 && stock[n - 1] == ' ') --n;
    return std::string(stock, stock + n);
}

struct Ev {
    uint32_t seq = 0;
    char type = 0;
    char side = 0;
    uint32_t shares = 0;
    uint32_t price = 0;
    uint64_t id = 0;
    uint64_t id2 = 0;
    int sym = -1;
    char stock[8]{};
};

struct Ord {
    char side = 0;
    uint32_t price = 0;
    uint32_t shares = 0;
};

Ev decode(uint32_t seq, const uint8_t* b, uint16_t len) {
    Ev e;
    e.seq = seq;
    if (len < 1) return e;
    e.type = static_cast<char>(b[0]);
    auto need = [&](int n) { return len >= n; };
    if (e.type == 'R' && need(19)) {
        std::memcpy(e.stock, b + 11, 8);
    } else if (e.type == 'A' && need(36)) {
        e.id = be64(b + 11);
        e.side = static_cast<char>(b[19]);
        e.shares = be32(b + 20);
        std::memcpy(e.stock, b + 24, 8);
        e.price = be32(b + 32);
    } else if (e.type == 'E' && need(36)) {
        e.side = static_cast<char>(b[19]);
        e.shares = be32(b + 20);
        e.price = be32(b + 24);
        std::memcpy(e.stock, b + 28, 8);
    } else if (e.type == 'X' && need(23)) {
        e.id = be64(b + 11);
        e.shares = be32(b + 19);
    } else if (e.type == 'D' && need(19)) {
        e.id = be64(b + 11);
    } else if (e.type == 'U' && need(35)) {
        e.id = be64(b + 11);
        e.id2 = be64(b + 19);
        e.shares = be32(b + 27);
        e.price = be32(b + 31);
    } else
        e.type = '?';
    return e;
}

struct Engine {
    std::map<std::string, int> sym_of;
    std::vector<std::string> names;
    struct Sym {
        std::vector<Ev> evs;
        std::map<uint64_t, Ord> live;
        std::map<uint32_t, std::map<uint64_t, uint32_t>> bid, ask;
        struct Snap {
            uint32_t seq = 0;
            std::map<uint64_t, Ord> orders;
        };
        std::vector<Snap> snaps;
        int since = 0;
    };
    std::vector<Sym> books;
    std::unordered_map<uint64_t, int> owner;
    // Survives deletes so historical log walks can attribute X/D/U to a symbol.
    std::unordered_map<uint64_t, int> home;
    std::vector<Ev> log;
    std::unordered_map<uint32_t, Ev> rewritten;
    uint32_t max_seq = 0;
    bool ready = false;

    int intern(const std::string& name) {
        auto it = sym_of.find(name);
        if (it != sym_of.end()) return it->second;
        int id = static_cast<int>(names.size());
        names.push_back(name);
        sym_of.emplace(name, id);
        books.emplace_back();
        books.back().snaps.push_back(Sym::Snap{0, {}});
        return id;
    }

    static std::map<uint32_t, std::map<uint64_t, uint32_t>>& side_levels(
        std::map<uint32_t, std::map<uint64_t, uint32_t>>& bid,
        std::map<uint32_t, std::map<uint64_t, uint32_t>>& ask, char side) {
        return side == 'B' ? bid : ask;
    }

    static void erase_id(std::map<uint64_t, Ord>& orders,
                         std::map<uint32_t, std::map<uint64_t, uint32_t>>& bid,
                         std::map<uint32_t, std::map<uint64_t, uint32_t>>& ask, uint64_t id) {
        auto it = orders.find(id);
        if (it == orders.end()) return;
        auto& levels = side_levels(bid, ask, it->second.side);
        auto pit = levels.find(it->second.price);
        if (pit != levels.end()) {
            pit->second.erase(id);
            if (pit->second.empty()) levels.erase(pit);
        }
        orders.erase(it);
    }

    static void consume(std::map<uint64_t, Ord>& orders,
                        std::map<uint32_t, std::map<uint64_t, uint32_t>>& bid,
                        std::map<uint32_t, std::map<uint64_t, uint32_t>>& ask, char side,
                        uint32_t shares) {
        auto& levels = side_levels(bid, ask, side);
        while (shares && !levels.empty()) {
            auto pit = side == 'B' ? std::prev(levels.end()) : levels.begin();
            auto& ids = pit->second;
            while (shares && !ids.empty()) {
                auto cur = ids.begin();
                auto oit = orders.find(cur->first);
                if (oit == orders.end()) {
                    ids.erase(cur);
                    continue;
                }
                if (shares >= cur->second) {
                    shares -= cur->second;
                    orders.erase(oit);
                    ids.erase(cur);
                } else {
                    cur->second -= shares;
                    oit->second.shares -= shares;
                    shares = 0;
                }
            }
            if (ids.empty()) levels.erase(pit);
        }
    }

    static void apply_one(std::map<uint64_t, Ord>& orders,
                          std::map<uint32_t, std::map<uint64_t, uint32_t>>& bid,
                          std::map<uint32_t, std::map<uint64_t, uint32_t>>& ask, const Ev& e) {
        if (e.type == 'A') {
            erase_id(orders, bid, ask, e.id);
            orders[e.id] = Ord{e.side, e.price, e.shares};
            side_levels(bid, ask, e.side)[e.price][e.id] = e.shares;
        } else if (e.type == 'E') {
            consume(orders, bid, ask, e.side, e.shares);
        } else if (e.type == 'X') {
            auto it = orders.find(e.id);
            if (it == orders.end()) return;
            if (e.shares >= it->second.shares)
                erase_id(orders, bid, ask, e.id);
            else {
                it->second.shares -= e.shares;
                side_levels(bid, ask, it->second.side)[it->second.price][e.id] = it->second.shares;
            }
        } else if (e.type == 'D') {
            erase_id(orders, bid, ask, e.id);
        } else if (e.type == 'U') {
            auto it = orders.find(e.id);
            if (it == orders.end()) return;
            Ord kept = it->second;
            erase_id(orders, bid, ask, e.id);
            erase_id(orders, bid, ask, e.id2);
            kept.price = e.price;
            kept.shares = e.shares;
            orders[e.id2] = kept;
            side_levels(bid, ask, kept.side)[kept.price][e.id2] = kept.shares;
        }
    }

    static void rebuild_levels(const std::map<uint64_t, Ord>& orders,
                               std::map<uint32_t, std::map<uint64_t, uint32_t>>& bid,
                               std::map<uint32_t, std::map<uint64_t, uint32_t>>& ask) {
        bid.clear();
        ask.clear();
        for (const auto& [id, o] : orders) {
            if (o.shares == 0) continue;
            side_levels(bid, ask, o.side)[o.price][id] = o.shares;
        }
    }

    void note_snap(int sym, uint32_t seq) {
        auto& b = books[sym];
        if (++b.since < kSnapEvery) return;
        b.snaps.push_back(Sym::Snap{seq, b.live});
        b.since = 0;
    }

    void merge_rewrites() {
        if (rewritten.empty()) return;
        std::map<uint32_t, Ev> merged;
        for (const Ev& ev : log) merged[ev.seq] = ev;
        for (const auto& [seq, ev] : rewritten) merged[seq] = ev;
        rewritten.clear();
        log.clear();
        log.reserve(merged.size());
        for (const auto& [seq, ev] : merged) {
            (void)seq;
            log.push_back(ev);
        }
    }

    void replay_all() {
        books.clear();
        names.clear();
        sym_of.clear();
        owner.clear();
        home.clear();
        max_seq = 0;
        for (const Ev& raw : log) play(raw);
        ready = true;
    }

    void apply_rewrites() {
        uint32_t earliest = 0xffffffffu;
        for (const auto& [seq, ev] : rewritten) {
            (void)ev;
            earliest = std::min(earliest, seq);
        }
        merge_rewrites();
        if (earliest == 0xffffffffu) {
            ready = true;
            return;
        }
        // Checkpointed path: restore each symbol from its nearest snap at or
        // before the earliest rewrite, then replay only events after that snap.
        if (!books.empty()) {
            std::vector<uint32_t> cut(books.size(), 0);
            bool ok = true;
            for (size_t s = 0; s < books.size(); ++s) {
                const Sym::Snap* best = nullptr;
                for (const auto& snap : books[s].snaps) {
                    if (snap.seq < earliest) best = &snap;
                }
                if (!best) {
                    ok = false;
                    break;
                }
                cut[s] = best->seq;
            }
            if (ok) {
                owner.clear();
                max_seq = 0;
                for (size_t s = 0; s < books.size(); ++s) {
                    auto& b = books[s];
                    std::vector<Sym::Snap> kept;
                    for (const auto& snap : b.snaps) {
                        if (snap.seq < earliest) kept.push_back(snap);
                    }
                    b.live = kept.back().orders;
                    rebuild_levels(b.live, b.bid, b.ask);
                    b.evs.clear();
                    b.snaps.swap(kept);
                    b.since = 0;
                    for (const auto& [id, o] : b.live) {
                        (void)o;
                        owner[id] = static_cast<int>(s);
                        home[id] = static_cast<int>(s);
                    }
                    max_seq = std::max(max_seq, cut[s]);
                }
                for (const Ev& raw : log) {
                    if (raw.type == 'R') {
                        if (raw.seq > earliest) play(raw);
                        continue;
                    }
                    // Resolve symbol without mutating; skip if at or before cut.
                    int sym = -1;
                    if (raw.type == 'A' || raw.type == 'E') {
                        auto it = sym_of.find(trim_stock(raw.stock));
                        if (it != sym_of.end()) sym = it->second;
                    } else {
                        auto it = owner.find(raw.id);
                        if (it != owner.end()) sym = it->second;
                    }
                    if (sym >= 0 && raw.seq <= cut[static_cast<size_t>(sym)]) continue;
                    play(raw);
                }
                ready = true;
                return;
            }
        }
        replay_all();
    }

    void play(Ev e) {
        max_seq = std::max(max_seq, e.seq);
        if (e.type == 'R') {
            intern(trim_stock(e.stock));
            return;
        }
        if (e.type == 'E') {
            auto found = sym_of.find(trim_stock(e.stock));
            if (found == sym_of.end()) return;
            int s = found->second;
            e.sym = s;
            books[s].evs.push_back(e);
            apply_one(books[s].live, books[s].bid, books[s].ask, e);
            note_snap(s, e.seq);
            return;
        }
        if (e.type == 'A') {
            int s = intern(trim_stock(e.stock));
            e.sym = s;
            owner[e.id] = s;
            home[e.id] = s;
            books[s].evs.push_back(e);
            apply_one(books[s].live, books[s].bid, books[s].ask, e);
            note_snap(s, e.seq);
            return;
        }
        auto it = owner.find(e.id);
        if (it == owner.end()) return;
        int s = it->second;
        if (e.type == 'U' && books[s].live.find(e.id) == books[s].live.end()) {
            owner.erase(e.id);
            return;
        }
        e.sym = s;
        if (e.type == 'U') {
            owner.erase(e.id);
            owner[e.id2] = s;
            home[e.id2] = s;
        } else if (e.type == 'D')
            owner.erase(e.id);
        else if (e.type == 'X') {
            auto live = books[s].live.find(e.id);
            uint32_t have = live == books[s].live.end() ? 0 : live->second.shares;
            if (e.shares >= have) owner.erase(e.id);
        }
        books[s].evs.push_back(e);
        apply_one(books[s].live, books[s].bid, books[s].ask, e);
        note_snap(s, e.seq);
    }

    void apply_blob(const uint8_t* p, size_t n) {
        size_t i = 0;
        bool monotonic = true;
        std::vector<Ev> batch;
        while (i + 6 <= n) {
            uint32_t seq = be32(p + i);
            uint16_t len = static_cast<uint16_t>((p[i + 4] << 8) | p[i + 5]);
            i += 6;
            if (i + len > n) _exit(2);
            Ev e = decode(seq, p + i, len);
            i += len;
            if (e.type == '?') continue;
            batch.push_back(e);
            if (ready && seq <= max_seq) monotonic = false;
        }
        if (!monotonic) {
            for (const Ev& e : batch) rewritten[e.seq] = e;
            apply_rewrites();
            return;
        }
        log.insert(log.end(), batch.begin(), batch.end());
        for (Ev e : batch) play(e);
        ready = true;
    }

    bool belongs(const Ev& e, int sym) const {
        if (e.type == 'R') return false;
        if (e.type == 'A' || e.type == 'E') {
            auto it = sym_of.find(trim_stock(e.stock));
            return it != sym_of.end() && it->second == sym;
        }
        auto it = home.find(e.id);
        return it != home.end() && it->second == sym;
    }

    std::map<uint64_t, Ord> orders_at(int sym, uint32_t q) {
        auto& b = books[sym];
        if (q >= max_seq) return b.live;
        const Sym::Snap* snap = &b.snaps[0];
        for (const auto& s : b.snaps)
            if (s.seq <= q) snap = &s;
        auto orders = snap->orders;
        std::map<uint32_t, std::map<uint64_t, uint32_t>> bid, ask;
        rebuild_levels(orders, bid, ask);
        // Walk the merged log so mid-log queries stay correct after a checkpoint
        // restore (per-symbol evs would miss events between older snaps and cut).
        for (const Ev& e : log) {
            if (e.seq <= snap->seq) continue;
            if (e.seq > q) break;
            if (!belongs(e, sym)) continue;
            apply_one(orders, bid, ask, e);
        }
        return orders;
    }

    std::string format_one(int sym, const std::map<uint64_t, Ord>& orders) {
        std::map<uint32_t, std::pair<uint64_t, uint64_t>> bids, asks;
        for (const auto& [id, o] : orders) {
            if (o.shares == 0) continue;
            auto& levels = o.side == 'B' ? bids : asks;
            auto it = levels.find(o.price);
            if (it == levels.end())
                levels.emplace(o.price, std::pair<uint64_t, uint64_t>{o.shares, id});
            else {
                it->second.first += o.shares;
                it->second.second = std::min(it->second.second, id);
            }
        }
        auto side = [](const std::map<uint32_t, std::pair<uint64_t, uint64_t>>& levels, bool desc) {
            if (levels.empty()) return std::string("-");
            std::vector<std::pair<uint32_t, std::pair<uint64_t, uint64_t>>> v(levels.begin(), levels.end());
            if (desc) std::reverse(v.begin(), v.end());
            std::string out;
            for (size_t i = 0; i < v.size(); ++i) {
                if (i) out.push_back(',');
                out += std::to_string(v[i].first);
                out.push_back(':');
                out += std::to_string(v[i].second.first);
                out.push_back(':');
                out += std::to_string(v[i].second.second);
            }
            return out;
        };
        return names[sym] + " B " + side(bids, true) + " A " + side(asks, false);
    }

    std::string format_name(const std::string& name, uint32_t seq) {
        auto it = sym_of.find(name);
        if (it == sym_of.end()) return name + " B - A -";
        return format_one(it->second, orders_at(it->second, seq));
    }

    void query(const std::vector<std::pair<std::string, uint32_t>>& qs, std::string& out) {
        for (const auto& [name, seq] : qs) {
            out += format_name(name, seq);
            out.push_back('\n');
        }
        out += "END\nREADY\n";
    }

    void snap(uint32_t seq, std::string& out) {
        std::vector<std::string> lines;
        for (int s = 0; s < static_cast<int>(names.size()); ++s) {
            std::string line = format_one(s, orders_at(s, seq));
            if (line.find(" B - A -") != std::string::npos) continue;
            lines.push_back(line);
        }
        std::sort(lines.begin(), lines.end());
        for (const auto& line : lines) {
            out += line;
            out.push_back('\n');
        }
        out += "END\nREADY\n";
    }
};
}  // namespace ref

// ---------------------------------------------------------------- fast engine
// One replay record per routed event, in seq order within its symbol.
struct Rec {
    uint8_t type;  // 'A','E','X','D','U'; 0 = tombstone (no-op)
    uint8_t side;  // 0 bid, 1 ask (A/E only)
    uint16_t pad;
    uint32_t shares;
    uint32_t pid;   // A/U: price id within the symbol
    uint32_t slot;  // A/X/D/U: order slot; E: deepest price touched by the last replay
    uint32_t slot2; // U: slot of the replacement id; E: footprint kind (kFoot*)
};
constexpr uint32_t kFootPrice = 0, kFootEmptied = 1, kFootNone = 2;

// Per-symbol price dictionary: stable price ids plus a rank order by price.
struct Dict {
    U64Map pid_of;
    struct Txt {
        char c[15];
        uint8_t n;
    };
    std::vector<Txt> ptxt;        // pid -> "price:" text
    std::vector<uint32_t> price;  // pid -> price
    std::vector<uint32_t> rank;   // pid -> rank (ascending price)
    std::vector<uint32_t> at;     // rank -> pid
    uint32_t nranked = 0;
    uint32_t epoch = 1;
    Dict() { pid_of.init(256); }
    uint32_t get(uint32_t px) {
        uint32_t cand = static_cast<uint32_t>(price.size());
        uint32_t p = pid_of.insert(px, cand);
        if (p == cand) {
            price.push_back(px);
            rank.push_back(kNil);
            Txt t{};
            char* e = put_u32(t.c, px);
            *e++ = ':';
            t.n = static_cast<uint8_t>(e - t.c);
            ptxt.push_back(t);
        }
        return p;
    }
    void rerank() {
        if (nranked == price.size()) return;
        std::vector<uint32_t> fresh;
        fresh.reserve(price.size() - nranked);
        for (uint32_t p = nranked; p < price.size(); ++p) fresh.push_back(p);
        std::sort(fresh.begin(), fresh.end(), [&](uint32_t a, uint32_t b) { return price[a] < price[b]; });
        std::vector<uint32_t> merged(at.size() + fresh.size());
        std::merge(at.begin(), at.end(), fresh.begin(), fresh.end(), merged.begin(),
                   [&](uint32_t a, uint32_t b) { return price[a] < price[b]; });
        at.swap(merged);
        for (uint32_t r = 0; r < at.size(); ++r) rank[at[r]] = r;
        nranked = static_cast<uint32_t>(price.size());
        ++epoch;
    }
};

// State of one order in a timeline (used for rewrite deltas and checkpoint patches).
struct DState {
    uint32_t live, side, pid, shares;
    bool operator==(const DState& b) const {
        if (live != b.live) return false;
        return !live || (side == b.side && pid == b.pid && shares == b.shares);
    }
};
struct DEnt {
    uint32_t slot;
    DState o, n;  // old timeline, new timeline
};

struct Book {
    struct O {
        uint32_t shares, prev, next, gen, pid;
        uint32_t side;
    };
    struct L {
        uint32_t head, tail, cnt, pad;
        uint64_t tot;
    };
    std::vector<O> o;
    std::vector<L> lv[2];
    std::vector<uint64_t> bits[2], sum[2];
    const Dict* d = nullptr;
    const uint64_t* rid = nullptr;
    struct IdTxt {
        char c[23];
        uint8_t n;  // 0 = not formatted yet
    };
    IdTxt* rtxt = nullptr;  // slot -> decimal raw id (filled lazily)
    uint32_t gen = 1, epoch = 0;
    bool foot_changed = false;  // some replayed execution moved its footprint

    bool live(uint32_t s) const { return o[s].gen == gen; }

    inline void set_bit(int s, uint32_t r) {
        bits[s][r >> 6] |= 1ull << (r & 63);
        sum[s][r >> 12] |= 1ull << ((r >> 6) & 63);
    }
    inline void clr_bit(int s, uint32_t r) {
        uint64_t& w = bits[s][r >> 6];
        w &= ~(1ull << (r & 63));
        if (!w) sum[s][r >> 12] &= ~(1ull << ((r >> 6) & 63));
    }
    // Best rank: highest for bids, lowest for asks; -1 if the side is empty.
    inline int64_t best(int s) const {
        const uint64_t* sm = sum[s].data();
        size_t n = sum[s].size();
        if (s == 0) {
            for (size_t i = n; i-- > 0;)
                if (sm[i]) {
                    size_t wi = i * 64 + 63 - __builtin_clzll(sm[i]);
                    return static_cast<int64_t>(wi * 64 + 63 - __builtin_clzll(bits[0][wi]));
                }
        } else {
            for (size_t i = 0; i < n; ++i)
                if (sm[i]) {
                    size_t wi = i * 64 + __builtin_ctzll(sm[i]);
                    return static_cast<int64_t>(wi * 64 + __builtin_ctzll(bits[1][wi]));
                }
        }
        return -1;
    }

    // Size arrays for the symbol's current slots/prices; rebuild rank bitmaps
    // if the price ranking changed since they were built.
    void prepare(const Dict& dict, const std::vector<uint64_t>& rids, std::vector<IdTxt>& rt) {
        d = &dict;
        rid = rids.data();
        if (rt.size() < rids.size()) rt.resize(rids.size() + 1024, IdTxt{{0}, 0});
        rtxt = rt.data();
        if (o.size() < rids.size()) o.resize(rids.size() + 1024, O{0, kNil, kNil, 0, 0, 0});
        size_t np = dict.price.size();
        if (lv[0].size() < np) {
            for (int s = 0; s < 2; ++s) lv[s].resize(np + 1024, L{kNil, kNil, 0, 0, 0});
        }
        size_t nw = (lv[0].size() + 63) / 64;
        size_t ns = (nw + 63) / 64;
        if (bits[0].size() < nw) {
            for (int s = 0; s < 2; ++s) {
                bits[s].resize(nw, 0);
                sum[s].resize(ns, 0);
            }
        }
        if (epoch != dict.epoch) {
            for (int s = 0; s < 2; ++s) {
                std::fill(bits[s].begin(), bits[s].end(), 0);
                std::fill(sum[s].begin(), sum[s].end(), 0);
                for (uint32_t p = 0; p < np; ++p)
                    if (lv[s][p].cnt) set_bit(s, dict.rank[p]);
            }
            epoch = dict.epoch;
        }
    }

    void insert(uint32_t s, uint32_t side, uint32_t pid, uint32_t shares) {
        L& l = lv[side][pid];
        O& x = o[s];
        x.shares = shares;
        x.pid = pid;
        x.side = side;
        x.gen = gen;
        if (l.cnt == 0) {
            l.head = l.tail = s;
            x.prev = x.next = kNil;
            l.cnt = 1;
            l.tot = shares;
            set_bit(static_cast<int>(side), d->rank[pid]);
            return;
        }
        uint64_t r = rid[s];
        if (rid[l.tail] < r) {
            x.prev = l.tail;
            x.next = kNil;
            o[l.tail].next = s;
            l.tail = s;
        } else {
            uint32_t c = l.tail;
            while (c != kNil && rid[c] > r) c = o[c].prev;
            x.prev = c;
            if (c == kNil) {
                x.next = l.head;
                o[l.head].prev = s;
                l.head = s;
            } else {
                x.next = o[c].next;
                o[x.next].prev = s;
                o[c].next = s;
            }
        }
        l.cnt++;
        l.tot += shares;
    }

    void remove(uint32_t s) {
        O& x = o[s];
        L& l = lv[x.side][x.pid];
        if (x.prev != kNil)
            o[x.prev].next = x.next;
        else
            l.head = x.next;
        if (x.next != kNil)
            o[x.next].prev = x.prev;
        else
            l.tail = x.prev;
        x.gen = 0;
        if (--l.cnt == 0) {
            l.tot = 0;
            clr_bit(static_cast<int>(x.side), d->rank[x.pid]);
        } else {
            l.tot -= x.shares;
        }
    }

    // Executes against the best levels; records how deep it reached in rec
    // (and flags the book when that footprint differs from the stored one).
    void consume(Rec& rec) {
        uint32_t shares = rec.shares;
        int s = rec.side;
        uint32_t fs = rec.slot, fk = rec.slot2;
        if (!shares) {
            rec.slot2 = kFootNone;
        } else {
            while (true) {
                int64_t r = best(s);
                if (r < 0) {
                    rec.slot2 = kFootEmptied;
                    break;
                }
                uint32_t pid = d->at[r];
                L& l = lv[s][pid];
                while (shares && l.cnt) {
                    uint32_t c = l.head;
                    O& x = o[c];
                    if (shares >= x.shares) {
                        shares -= x.shares;
                        l.head = x.next;
                        if (l.head != kNil) o[l.head].prev = kNil;
                        l.cnt--;
                        l.tot -= x.shares;
                        x.gen = 0;
                    } else {
                        x.shares -= shares;
                        l.tot -= shares;
                        shares = 0;
                    }
                }
                if (l.cnt == 0) {
                    l.tot = 0;
                    clr_bit(s, static_cast<uint32_t>(r));
                }
                if (!shares) {
                    rec.slot = d->price[pid];
                    rec.slot2 = kFootPrice;
                    break;
                }
            }
        }
        if (rec.slot2 != fk || (fk == kFootPrice && rec.slot != fs)) foot_changed = true;
    }

    inline void apply(Rec& r) {
        switch (r.type) {
            case 'A':
                if (live(r.slot)) remove(r.slot);
                insert(r.slot, r.side, r.pid, r.shares);
                break;
            case 'E':
                consume(r);
                break;
            case 'X':
                if (live(r.slot)) {
                    O& x = o[r.slot];
                    if (r.shares >= x.shares) {
                        remove(r.slot);
                    } else {
                        x.shares -= r.shares;
                        lv[x.side][x.pid].tot -= r.shares;
                    }
                }
                break;
            case 'D':
                if (live(r.slot)) remove(r.slot);
                break;
            case 'U':
                if (live(r.slot)) {
                    uint32_t side = o[r.slot].side;
                    remove(r.slot);
                    if (live(r.slot2)) remove(r.slot2);
                    insert(r.slot2, side, r.pid, r.shares);
                }
                break;
            default:
                break;
        }
    }

    void patch(const DEnt& e) {
        if (e.o.live) remove(e.slot);
        if (e.n.live) insert(e.slot, e.n.side, e.n.pid, e.n.shares);
    }

    template <class F>
    void each_desc(int s, F f) const {
        for (size_t i = sum[s].size(); i-- > 0;) {
            uint64_t sw = sum[s][i];
            while (sw) {
                int b = 63 - __builtin_clzll(sw);
                sw &= ~(1ull << b);
                size_t wi = i * 64 + b;
                uint64_t w = bits[s][wi];
                while (w) {
                    int c = 63 - __builtin_clzll(w);
                    w &= ~(1ull << c);
                    f(static_cast<uint32_t>(wi * 64 + c));
                }
            }
        }
    }
    template <class F>
    void each_asc(int s, F f) const {
        for (size_t i = 0; i < sum[s].size(); ++i) {
            uint64_t sw = sum[s][i];
            while (sw) {
                int b = __builtin_ctzll(sw);
                sw &= sw - 1;
                size_t wi = i * 64 + b;
                uint64_t w = bits[s][wi];
                while (w) {
                    int c = __builtin_ctzll(w);
                    w &= w - 1;
                    f(static_cast<uint32_t>(wi * 64 + c));
                }
            }
        }
    }

    void clear() {
        for (int s = 0; s < 2; ++s) {
            for (size_t i = 0; i < sum[s].size(); ++i) {
                uint64_t sw = sum[s][i];
                while (sw) {
                    int b = __builtin_ctzll(sw);
                    sw &= sw - 1;
                    size_t wi = i * 64 + b;
                    uint64_t w = bits[s][wi];
                    while (w) {
                        int c = __builtin_ctzll(w);
                        w &= w - 1;
                        lv[s][d->at[wi * 64 + c]].cnt = 0;
                    }
                    bits[s][wi] = 0;
                }
                sum[s][i] = 0;
            }
        }
        if (++gen == 0) {
            for (auto& x : o) x.gen = 0;
            gen = 1;
        }
    }

    // Checkpoint layout: n_levels, then per level: side<<31|pid, cnt, cnt x (slot, shares).
    void save(std::vector<uint32_t>& ck) const {
        size_t need = 1;
        uint32_t nl = 0;
        for (int s = 0; s < 2; ++s)
            each_asc(s, [&](uint32_t r) {
                need += 2 + 2 * lv[s][d->at[r]].cnt;
                ++nl;
            });
        ck.resize(need);
        uint32_t* p = ck.data();
        *p++ = nl;
        for (int s = 0; s < 2; ++s)
            each_asc(s, [&](uint32_t r) {
                uint32_t pid = d->at[r];
                const L& l = lv[s][pid];
                *p++ = static_cast<uint32_t>(s) << 31 | pid;
                *p++ = l.cnt;
                for (uint32_t c = l.head; c != kNil; c = o[c].next) {
                    *p++ = c;
                    *p++ = o[c].shares;
                }
            });
    }

    void load(const std::vector<uint32_t>& ck) {
        clear();
        const uint32_t* p = ck.data();
        uint32_t nl = *p++;
        for (uint32_t i = 0; i < nl; ++i) {
            uint32_t h = *p++;
            uint32_t s = h >> 31, pid = h & 0x7fffffffu;
            uint32_t cnt = *p++;
            L& l = lv[s][pid];
            l.cnt = cnt;
            uint64_t tot = 0;
            uint32_t prev = kNil;
            for (uint32_t c = 0; c < cnt; ++c) {
                uint32_t sl = *p++;
                uint32_t sh = *p++;
                O& x = o[sl];
                x.shares = sh;
                x.pid = pid;
                x.side = s;
                x.gen = gen;
                x.prev = prev;
                x.next = kNil;
                if (prev != kNil)
                    o[prev].next = sl;
                else
                    l.head = sl;
                prev = sl;
                tot += sh;
            }
            l.tail = prev;
            l.tot = tot;
            set_bit(static_cast<int>(s), d->rank[pid]);
        }
    }

    // Appends "<name> B ... A ..." (no newline).
    void format(const std::string& name) const {
        size_t nl = 0;
        for (int s = 0; s < 2; ++s)
            for (uint64_t w : bits[s]) nl += static_cast<size_t>(__builtin_popcountll(w));
        out.reserve(name.size() + 64 + nl * 64);
        char* p = out.p + out.n;
        memcpy(p, name.data(), name.size());
        p += name.size();
        for (int s = 0; s < 2; ++s) {
            memcpy(p, s ? " A " : " B ", 3);
            p += 3;
            char* start = p;
            auto emit = [&](uint32_t r) {
                uint32_t pid = d->at[r];
                const L& l = lv[s][pid];
                if (l.tot == 0) return;
                uint32_t c = l.head;
                while (o[c].shares == 0) c = o[c].next;
                *p = ',';
                p += p != start;
                const Dict::Txt& pt = d->ptxt[pid];
                memcpy(p, pt.c, 16);  // fixed-size copy; output has slack
                p += pt.n;
                p = put_u64(p, l.tot);
                *p++ = ':';
                IdTxt& it = rtxt[c];
                if (!it.n) it.n = static_cast<uint8_t>(put_u64(it.c, rid[c]) - it.c);
                memcpy(p, it.c, 24);
                p += it.n;
            };
            if (s == 0)
                each_desc(0, emit);
            else
                each_asc(1, emit);
            if (p == start) *p++ = '-';
        }
        out.n = static_cast<size_t>(p - out.p);
    }
};

constexpr uint32_t kCk = 256;  // records between checkpoints
constexpr uint32_t kBlk = 64;  // records per summary block (divides kCk)

// Conservative digest of a block of records: which order slots it may touch
// and how deep its executions reached on each side (from stored footprints).
struct BlkSum {
    uint64_t bloom[4];
    int64_t bid_lo;  // executions reach bids priced >= bid_lo
    int64_t ask_hi;  // executions reach asks priced <= ask_hi
    uint32_t ok;
};
inline uint32_t bloom_bit(uint32_t slot) { return (slot * 0x9E3779B1u) >> 24; }

struct Sym {
    uint64_t key = 0;
    std::string name;
    std::vector<uint32_t> seqs;  // parallel to recs
    std::vector<Rec> recs;
    std::vector<std::pair<uint32_t, Rec>> pending;
    std::vector<uint64_t> rid;  // slot -> raw id
    U64Map slot_of;             // dense id -> slot, for ids routed elsewhere now
    std::vector<Book::IdTxt> rtxt;
    Dict dict;
    // cks[m]: book after the first m*kCk records, plus patches applied in order.
    // Valid while m*kCk <= vend; E footprints are valid for records below vend.
    std::vector<std::vector<uint32_t>> cks;
    std::vector<std::vector<DEnt>> cpatch;
    uint32_t vend = 0;
    Book book;
    uint32_t pos = 0;  // book holds the state after recs[0..pos) when bvalid
    bool bvalid = true;
    std::vector<BlkSum> bsum;
    // Second book for reads far from `book` (keeps `book` at the tip/speculation).
    Book scratch;
    uint32_t spos = 0;
    bool svalid = false;

    void inval(size_t lo, size_t hi) {
        if (hi <= lo) return;
        size_t b1 = std::min(bsum.size(), (hi - 1) / kBlk + 1);
        for (size_t b = lo / kBlk; b < b1; ++b) bsum[b].ok = 0;
    }
    const BlkSum& summary(uint32_t b) {
        if (bsum.size() <= b) bsum.resize(b + 1, BlkSum{{0, 0, 0, 0}, INT64_MAX, -1, 0});
        BlkSum& s = bsum[b];
        if (s.ok) return s;
        s = BlkSum{{0, 0, 0, 0}, INT64_MAX, -1, 1};
        size_t e = std::min(recs.size(), static_cast<size_t>(b + 1) * kBlk);
        for (size_t i = static_cast<size_t>(b) * kBlk; i < e; ++i) {
            const Rec& r = recs[i];
            switch (r.type) {
                case 'U': {
                    uint32_t k = bloom_bit(r.slot2);
                    s.bloom[k >> 6] |= 1ull << (k & 63);
                }
                    // fallthrough
                case 'A':
                case 'X':
                case 'D': {
                    uint32_t k = bloom_bit(r.slot);
                    s.bloom[k >> 6] |= 1ull << (k & 63);
                    break;
                }
                case 'E':
                    if (r.slot2 == kFootPrice) {
                        if (r.side == 0)
                            s.bid_lo = std::min<int64_t>(s.bid_lo, r.slot);
                        else
                            s.ask_hi = std::max<int64_t>(s.ask_hi, r.slot);
                    } else if (r.slot2 == kFootEmptied) {
                        if (r.side == 0)
                            s.bid_lo = 0;
                        else
                            s.ask_hi = INT64_MAX;
                    }
                    break;
                default:
                    break;
            }
        }
        return s;
    }

    uint32_t new_slot(uint64_t raw) {
        rid.push_back(raw);
        return static_cast<uint32_t>(rid.size() - 1);
    }
    void dirty(uint32_t idx) {
        if (idx < vend) vend = idx;
        if (bvalid && pos > idx) bvalid = false;
        if (svalid && spos > idx) svalid = false;
    }
    size_t lower(uint32_t seq) const {
        return static_cast<size_t>(std::lower_bound(seqs.begin(), seqs.end(), seq) - seqs.begin());
    }
    size_t upper(uint32_t seq) const {
        return static_cast<size_t>(std::upper_bound(seqs.begin(), seqs.end(), seq) - seqs.begin());
    }
};

struct Fast {
    std::vector<Sym> syms;
    U64Map sym_of;
    std::vector<Ev> evs;
    std::vector<uint32_t> gseq, gidx;  // sorted seqs -> event index
    U64Map id_of;                      // raw id -> dense id
    std::vector<uint64_t> id_raw;
    struct IdInfo {
        int32_t sym;    // symbol its X/D/U events route to (-1 none)
        uint32_t slot;  // its order slot in that symbol
        uint32_t head;  // first event naming it: (evidx << 1 | field)
        uint32_t conf;  // added in more than one symbol
    };
    std::vector<IdInfo> idi;
    // Direct window for ids near the first one seen (stores dense id + 1).
    static constexpr uint64_t kWin = 1ull << 20;
    uint32_t* win = nullptr;
    uint64_t win_base = 0;
    uint32_t max_seq = 0;
    bool ready = false;
    bool weird = false;
    // Some id is added in several symbols.  Below-tip answers then follow the
    // reference's walk (X/D routed by the id's last add); tip answers need the
    // reference itself.
    bool has_conflict = false;
    std::vector<int32_t> touched;
    std::vector<DEnt> delta;
    // A rewrite whose delta has not been propagated yet (done after READY).
    int32_t pend_sym = -1;
    uint32_t pend_i = 0;
    std::vector<DEnt> pend_delta;

    Fast() {
        sym_of.init(64);
        id_of.init(64);  // most ids go through the direct window
    }

    uint64_t last_stock = 0;
    int32_t last_sym = -1;
    int32_t intern(uint64_t stock) {
        if (stock == last_stock && last_sym >= 0) return last_sym;
        int32_t r = intern_slow(stock);
        last_stock = stock;
        last_sym = r;
        return r;
    }
    int32_t intern_slow(uint64_t stock) {
        uint32_t s = sym_of.find(stock);
        if (s != U64Map::kNone) return static_cast<int32_t>(s);
        s = static_cast<uint32_t>(syms.size());
        sym_of.insert(stock, s);
        syms.emplace_back();
        Sym& y = syms.back();
        y.key = stock;
        char nm[8];
        memcpy(nm, &stock, 8);
        int n = 8;
        while (n > 0 && nm[n - 1] == ' ') --n;
        y.name.assign(nm, nm + n);
        y.cks.emplace_back(std::vector<uint32_t>{0});
        y.cpatch.emplace_back();
        return static_cast<int32_t>(s);
    }

    uint32_t new_dense(uint64_t id) {
        uint32_t d = static_cast<uint32_t>(id_raw.size());
        id_raw.push_back(id);
        idi.push_back(IdInfo{-1, kNil, kNil, 0});
        return d;
    }
    uint32_t dense(uint64_t id) {
        if (!win) {
            void* m = mmap(nullptr, kWin * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            win = m == MAP_FAILED ? reinterpret_cast<uint32_t*>(1) : static_cast<uint32_t*>(m);
            win_base = id - std::min<uint64_t>(id, kWin / 8);
        }
        uint64_t off = id - win_base;
        if (off < kWin && win != reinterpret_cast<uint32_t*>(1)) {
            uint32_t& w = win[off];
            if (w) return w - 1;
            uint32_t d = new_dense(id);
            w = d + 1;
            return d;
        }
        uint32_t cand = static_cast<uint32_t>(id_raw.size());
        uint32_t d = id_of.insert(id, cand);
        if (d != cand) return d;
        return new_dense(id);
    }

    uint32_t find_seq(uint32_t seq) const {
        auto it = std::lower_bound(gseq.begin(), gseq.end(), seq);
        if (it == gseq.end() || *it != seq) return kNil;
        return gidx[static_cast<size_t>(it - gseq.begin())];
    }

    void ref_push(uint32_t idx, int k) {
        Ev& e = evs[idx];
        uint32_t d = k ? e.d2 : e.d;
        e.nref[k] = idi[d].head;
        idi[d].head = idx << 1 | static_cast<uint32_t>(k);
    }
    void ref_remove(uint32_t idx, int k) {
        uint32_t d = k ? evs[idx].d2 : evs[idx].d;
        uint32_t want = idx << 1 | static_cast<uint32_t>(k);
        uint32_t* link = &idi[d].head;
        while (*link != kNil) {
            if (*link == want) {
                *link = evs[idx].nref[k];
                return;
            }
            uint32_t cur = *link;
            link = &evs[cur >> 1].nref[cur & 1];
        }
    }

    // Slot of dense id d in symbol s (stable for the life of the session).
    uint32_t slot_in(int32_t s, uint32_t d) {
        if (idi[d].sym == s && idi[d].slot != kNil) return idi[d].slot;
        Sym& y = syms[s];
        uint32_t cand = static_cast<uint32_t>(y.rid.size());
        uint32_t v = y.slot_of.insert(d, cand);
        if (v == cand) y.rid.push_back(id_raw[d]);
        return v;
    }

    Rec make_rec(const Ev& e, Sym& y) {
        Rec r{};
        r.type = static_cast<uint8_t>(e.type);
        r.side = e.side == 'B' ? 0 : 1;
        r.shares = e.shares;
        switch (e.type) {
            case 'A':
                r.pid = y.dict.get(e.price);
                r.slot = slot_in(e.rsym, e.d);
                break;
            case 'X':
            case 'D':
                r.slot = idi[e.d].slot;
                break;
            case 'U':
                r.pid = y.dict.get(e.price);
                r.slot = idi[e.d].slot;
                r.slot2 = idi[e.d2].slot;
                break;
            default:
                break;
        }
        if (r.slot == kNil || r.slot2 == kNil) {
            weird = true;
            r.type = 0;
            r.slot = r.slot2 = 0;
        }
        return r;
    }

    void insert_rec(int32_t s, const Ev& e) {
        Sym& y = syms[s];
        Rec r = make_rec(e, y);
        if (y.seqs.empty() || e.seq > y.seqs.back()) {
            if (y.pending.empty()) {
                y.seqs.push_back(e.seq);
                y.recs.push_back(r);
                y.inval(y.recs.size() - 1, y.recs.size());
                return;
            }
        } else {
            size_t i = y.lower(e.seq);
            if (i < y.seqs.size() && y.seqs[i] == e.seq) {
                y.recs[i] = r;  // reuse the tombstone left at this seq
                y.inval(i, i + 1);
                y.dirty(static_cast<uint32_t>(i));
                return;
            }
        }
        if (y.pending.empty()) touched.push_back(s);
        y.pending.push_back({e.seq, r});
    }

    void remove_rec(int32_t s, uint32_t seq) {
        Sym& y = syms[s];
        size_t i = y.lower(seq);
        if (i < y.seqs.size() && y.seqs[i] == seq) {
            if (y.recs[i].type != 0) {
                y.recs[i].type = 0;
                y.inval(i, i + 1);
                y.dirty(static_cast<uint32_t>(i));
            }
            return;
        }
        for (size_t k = 0; k < y.pending.size(); ++k)
            if (y.pending[k].first == seq) {
                y.pending.erase(y.pending.begin() + static_cast<long>(k));
                return;
            }
    }

    void merge_pending() {
        for (int32_t s : touched) {
            Sym& y = syms[s];
            if (y.pending.empty()) continue;
            std::sort(y.pending.begin(), y.pending.end(),
                      [](const std::pair<uint32_t, Rec>& a, const std::pair<uint32_t, Rec>& b) { return a.first < b.first; });
            size_t first = y.lower(y.pending.front().first);
            size_t n = y.seqs.size(), m = y.pending.size();
            y.seqs.resize(n + m);
            y.recs.resize(n + m);
            size_t a = n, b = m, w = n + m;
            while (b > 0) {
                if (a > first && y.seqs[a - 1] > y.pending[b - 1].first) {
                    --a;
                    --w;
                    y.seqs[w] = y.seqs[a];
                    y.recs[w] = y.recs[a];
                } else {
                    --b;
                    --w;
                    y.seqs[w] = y.pending[b].first;
                    y.recs[w] = y.pending[b].second;
                }
            }
            y.pending.clear();
            y.inval(first, y.recs.size());
            y.dirty(static_cast<uint32_t>(first));
        }
        touched.clear();
    }

    // Symbol X/D/U events of dense id d route to: that of its last add (or
    // replacement) in seq order, -1 if none.  Ids added in several symbols are
    // reproduced only when no replacement touches them.
    int32_t compute_sym(uint32_t d) {
        int32_t best = -1, first = -1;
        uint32_t best_seq = 0;
        bool multi = false, any_u = false;
        for (uint32_t cur = idi[d].head; cur != kNil;) {
            const Ev& e = evs[cur >> 1];
            int k = static_cast<int>(cur & 1);
            int32_t c = -1;
            if (e.type == 'A' && k == 0) {
                c = e.rsym;
            } else if (e.type == 'U') {
                any_u = true;
                if (k == 1) c = e.rsym;
            }
            if (c >= 0) {
                if (first < 0)
                    first = c;
                else if (c != first)
                    multi = true;
                if (best < 0 || e.seq > best_seq) {
                    best = c;
                    best_seq = e.seq;
                }
            }
            cur = e.nref[k];
        }
        idi[d].conf = multi;
        if (multi) {
            has_conflict = true;
            if (any_u) weird = true;
        }
        return best;
    }

    void reroute(uint32_t d, int depth = 0) {
        if (depth > 1000) {
            weird = true;
            return;
        }
        int32_t ns = compute_sym(d);
        if (ns == idi[d].sym) return;
        int32_t os = idi[d].sym;
        if (os >= 0 && idi[d].slot != kNil) syms[os].slot_of.insert(d, idi[d].slot);
        idi[d].sym = ns;
        if (ns >= 0) {
            uint32_t v = syms[ns].slot_of.find(d);
            idi[d].slot = v != U64Map::kNone ? v : syms[ns].new_slot(id_raw[d]);
        } else {
            idi[d].slot = kNil;
        }
        std::vector<uint32_t> mv;
        for (uint32_t cur = idi[d].head; cur != kNil;) {
            const Ev& e = evs[cur >> 1];
            int k = static_cast<int>(cur & 1);
            if (k == 0 && (e.type == 'X' || e.type == 'D' || e.type == 'U')) mv.push_back(cur >> 1);
            cur = e.nref[k];
        }
        for (uint32_t idx : mv) {
            Ev& e = evs[idx];
            if (e.rsym >= 0) remove_rec(e.rsym, e.seq);
            e.rsym = ns;
            if (e.type == 'U') reroute(evs[idx].d2, depth + 1);
            if (ns >= 0) insert_rec(ns, evs[idx]);
        }
    }

    void unlink_event(uint32_t idx) {
        Ev old = evs[idx];
        if (old.rsym >= 0) remove_rec(old.rsym, old.seq);
        evs[idx].rsym = -1;
        if (old.d != kNil) ref_remove(idx, 0);
        if (old.d2 != kNil) ref_remove(idx, 1);
        evs[idx].type = 0;
        if (old.type == 'A') reroute(old.d);
        if (old.type == 'U') reroute(old.d2);
    }

    void link_event(uint32_t idx) {
        Ev& e = evs[idx];
        switch (e.type) {
            case 'R':
                intern(e.stock);
                break;
            case 'A': {
                int32_t s = intern(e.stock);
                Ev& a = evs[idx];
                a.d = dense(a.id);
                a.rsym = s;
                ref_push(idx, 0);
                uint32_t d = a.d;
                if (idi[d].sym != s) {
                    if (idi[d].sym < 0 && evs[idx].nref[0] == kNil) {
                        idi[d].sym = s;  // fresh id: no other events to move
                        idi[d].slot = syms[s].new_slot(evs[idx].id);
                    } else {
                        reroute(d);
                    }
                }
                insert_rec(s, evs[idx]);
                break;
            }
            case 'E':
                e.rsym = intern(e.stock);
                insert_rec(e.rsym, e);
                break;
            case 'X':
            case 'D': {
                e.d = dense(e.id);
                Ev& x = evs[idx];
                ref_push(idx, 0);
                x.rsym = idi[x.d].sym;
                if (x.rsym >= 0) insert_rec(x.rsym, x);
                break;
            }
            case 'U': {
                e.d = dense(e.id);
                evs[idx].d2 = dense(evs[idx].id2);
                Ev& u = evs[idx];
                ref_push(idx, 0);
                ref_push(idx, 1);
                if (idi[u.d].conf || idi[u.d2].conf) weird = true;
                u.rsym = idi[u.d].sym;
                uint32_t d2 = u.d2;
                if (u.rsym >= 0 && idi[d2].sym != u.rsym) {
                    if (idi[d2].sym < 0 && evs[idx].nref[1] == kNil) {
                        idi[d2].sym = u.rsym;
                        idi[d2].slot = syms[u.rsym].new_slot(evs[idx].id2);
                    } else {
                        reroute(d2);
                    }
                }
                if (evs[idx].rsym >= 0) insert_rec(evs[idx].rsym, evs[idx]);
                break;
            }
            default:
                break;
        }
    }

    template <class V>
    static void grow_for(V& v, size_t extra) {
        if (v.capacity() < v.size() + extra) v.reserve(std::max(v.capacity() * 2, v.size() + extra));
    }
    void reserve_for(size_t k) {
        grow_for(evs, k);
        grow_for(gseq, k);
        grow_for(gidx, k);
        grow_for(id_raw, k);
        grow_for(idi, k);
    }

    // Appends an event with seq above everything seen.
    void append(const Ev& e) {
        settle();
        uint32_t idx = static_cast<uint32_t>(evs.size());
        evs.push_back(e);
        gseq.push_back(e.seq);
        gidx.push_back(idx);
        if (e.seq == 0 && e.type != 'R' && e.type != 0) weird = true;
        link_event(idx);
        if (e.seq > max_seq) max_seq = e.seq;
    }

    // ---- rewrite deltas
    int dfind(uint32_t slot) const {
        for (size_t k = 0; k < delta.size(); ++k)
            if (delta[k].slot == slot) return static_cast<int>(k);
        return -1;
    }
    void dprune() {
        size_t w = 0;
        for (size_t k = 0; k < delta.size(); ++k)
            if (!(delta[k].o == delta[k].n)) delta[w++] = delta[k];
        delta.resize(w);
    }
    static bool reaches(const Dict& dict, const DState& st, const Rec& e) {
        if (!st.live || st.side != e.side || e.slot2 == kFootNone) return false;
        if (e.slot2 == kFootEmptied) return true;
        uint32_t px = dict.price[st.pid];
        return e.side == 0 ? px >= e.slot : px <= e.slot;
    }
    // Advances the delta over one record; false if the effect cannot be tracked.
    bool dstep(const Sym& y, const Rec& x) {
        switch (x.type) {
            case 'A':
            case 'D': {
                int k = dfind(x.slot);
                if (k >= 0) delta.erase(delta.begin() + k);  // both timelines now agree
                return true;
            }
            case 'X': {
                int k = dfind(x.slot);
                if (k < 0) return true;
                for (DState* st : {&delta[k].o, &delta[k].n}) {
                    if (!st->live) continue;
                    if (x.shares >= st->shares)
                        st->live = 0;
                    else
                        st->shares -= x.shares;
                }
                dprune();
                return true;
            }
            case 'U': {
                int k = dfind(x.slot);
                int k2 = dfind(x.slot2);
                if (k < 0) return k2 < 0;
                DState so = delta[k].o, sn = delta[k].n;
                bool eo = so.live, en = sn.live;
                if (!eo && !en) return true;
                if (x.slot2 == x.slot) {
                    if (eo) delta[k].o = DState{1, so.side, x.pid, x.shares};
                    if (en) delta[k].n = DState{1, sn.side, x.pid, x.shares};
                    dprune();
                    return true;
                }
                DState o2, n2;
                if (k2 >= 0) {
                    o2 = delta[k2].o;
                    n2 = delta[k2].n;
                } else if (eo && en) {
                    o2 = n2 = DState{0, 0, 0, 0};  // overwritten below
                } else {
                    return false;  // replacement id's prior state is unknown
                }
                if (eo) o2 = DState{1, so.side, x.pid, x.shares};
                if (en) n2 = DState{1, sn.side, x.pid, x.shares};
                delta[k].o.live = 0;
                delta[k].n.live = 0;
                if (k2 >= 0) {
                    delta[k2].o = o2;
                    delta[k2].n = n2;
                } else {
                    delta.push_back(DEnt{x.slot2, o2, n2});
                }
                dprune();
                return true;
            }
            case 'E':
                for (const DEnt& e : delta)
                    if (reaches(y.dict, e.o, x) || reaches(y.dict, e.n, x)) return false;
                return true;
            default:
                return true;
        }
    }
    bool touches(const Sym& y, const BlkSum& bs) const {
        for (const DEnt& d : delta) {
            uint32_t k = bloom_bit(d.slot);
            if (bs.bloom[k >> 6] >> (k & 63) & 1) return true;
            for (const DState* st : {&d.o, &d.n}) {
                if (!st->live) continue;
                int64_t px = y.dict.price[st->pid];
                if (st->side == 0 ? px >= bs.bid_lo : px <= bs.ask_hi) return true;
            }
        }
        return false;
    }

    DState bstate(const Book& b, uint32_t slot) const {
        if (!b.live(slot)) return DState{0, 0, 0, 0};
        const Book::O& x = b.o[slot];
        return DState{1, x.side, x.pid, x.shares};
    }

    struct XO {
        uint64_t key;  // priority within the side (best first), then raw id
        uint64_t rid;
        uint32_t slot, shares, pid;
        uint32_t post;  // shares after the execution; kNil = consumed
    };
    std::vector<XO> xnew, xold;

    static void simulate(std::vector<XO>& v, uint32_t shares) {
        for (XO& x : v) x.post = x.shares;
        for (XO& x : v) {
            if (!shares) break;
            if (shares >= x.shares) {
                shares -= x.shares;
                x.post = kNil;
            } else {
                x.post = x.shares - shares;
                shares = 0;
            }
        }
    }

    // Exact delta update for record x, using the book positioned right before x
    // in the new timeline; then applies x to the book.
    void resolve(Sym& y, Rec& x) {
        Book& b = y.book;
        if (x.type == 'U') {
            int k = dfind(x.slot), k2 = dfind(x.slot2);
            if (k < 0) {
                // Same order in both timelines: the replacement agrees iff it happens.
                if (b.live(x.slot) && k2 >= 0) delta.erase(delta.begin() + k2);
            } else {
                DState prior = bstate(b, x.slot2);
                DState so = delta[k].o, sn = delta[k].n;
                DState o2 = so.live ? DState{1, so.side, x.pid, x.shares} : prior;
                DState n2 = sn.live ? DState{1, sn.side, x.pid, x.shares} : prior;
                delta[k].o.live = 0;
                delta[k].n.live = 0;
                if (k2 >= 0) {
                    delta[k2].o = o2;
                    delta[k2].n = n2;
                } else {
                    delta.push_back(DEnt{x.slot2, o2, n2});
                }
                dprune();
            }
            b.apply(x);
            return;
        }
        // Execution: replay it on the top of the side in both timelines.
        uint32_t s = x.side;
        uint64_t need = x.shares;
        for (const DEnt& d : delta)
            if (d.n.live && d.n.side == s) need += d.n.shares;
        auto prio = [&](uint32_t pid) -> uint64_t {
            uint32_t r = y.dict.rank[pid];
            return s == 0 ? static_cast<uint64_t>(0xffffffffu - r) : r;
        };
        xnew.clear();
        uint64_t cum = 0;
        int64_t r = b.best(static_cast<int>(s));
        while (r >= 0 && cum < need) {
            uint32_t pid = y.dict.at[r];
            uint64_t pk = prio(pid);
            for (uint32_t c = b.lv[s][pid].head; c != kNil; c = b.o[c].next) {
                xnew.push_back(XO{pk, y.rid[c], c, b.o[c].shares, pid, 0});
                cum += b.o[c].shares;
            }
            // next level in priority order
            int64_t nr = -1;
            if (s == 0) {
                for (int64_t q = r - 1; q >= 0; --q) {
                    uint64_t w = b.bits[0][q >> 6] & (~0ull >> (63 - (q & 63)));
                    if (w) {
                        nr = (q & ~63ll) + 63 - __builtin_clzll(w);
                        break;
                    }
                    q &= ~63ll;
                }
            } else {
                size_t nbits = b.bits[1].size() * 64;
                for (uint64_t q = static_cast<uint64_t>(r) + 1; q < nbits;) {
                    uint64_t w = b.bits[1][q >> 6] & (~0ull << (q & 63));
                    if (w) {
                        nr = static_cast<int64_t>((q & ~63ull) + __builtin_ctzll(w));
                        break;
                    }
                    q = (q | 63) + 1;
                }
            }
            r = nr;
        }
        xold = xnew;
        for (const DEnt& d : delta) {
            if (d.n.live && d.n.side == s)
                for (size_t k = 0; k < xold.size(); ++k)
                    if (xold[k].slot == d.slot) {
                        xold.erase(xold.begin() + static_cast<long>(k));
                        break;
                    }
            if (d.o.live && d.o.side == s) {
                XO add{prio(d.o.pid), y.rid[d.slot], d.slot, d.o.shares, d.o.pid, 0};
                size_t k = 0;
                while (k < xold.size() && (xold[k].key < add.key || (xold[k].key == add.key && xold[k].rid < add.rid))) ++k;
                xold.insert(xold.begin() + static_cast<long>(k), add);
            }
        }
        simulate(xnew, x.shares);
        simulate(xold, x.shares);
        auto post_of = [&](const XO& e) -> DState {
            if (e.post == kNil) return DState{0, 0, 0, 0};
            return DState{1, s, e.pid, e.post};
        };
        // Old/new states after the execution for every order it could touch.
        std::vector<DEnt> nd;
        for (const DEnt& d : delta) {
            bool in_new = false, in_old = false;
            for (const XO& e : xnew) in_new |= e.slot == d.slot;
            for (const XO& e : xold) in_old |= e.slot == d.slot;
            if (!in_new && !in_old) nd.push_back(d);
        }
        for (const XO& e : xold) {
            DEnt ent{e.slot, post_of(e), DState{0, 0, 0, 0}};
            int k = dfind(e.slot);
            bool found = false;
            for (const XO& f : xnew)
                if (f.slot == e.slot) {
                    ent.n = post_of(f);
                    found = true;
                    break;
                }
            if (!found) ent.n = delta[k].n;  // only differing orders can be missing from one side
            if (!(ent.o == ent.n)) nd.push_back(ent);
        }
        for (const XO& f : xnew) {
            bool in_old = false;
            for (const XO& e : xold) in_old |= e.slot == f.slot;
            if (in_old) continue;
            DEnt ent{f.slot, delta[dfind(f.slot)].o, post_of(f)};
            if (!(ent.o == ent.n)) nd.push_back(ent);
        }
        delta.swap(nd);
        b.apply(x);
    }

    // Propagates `delta` (state difference right after record i) through the
    // validated records, patching checkpoints and the live book on the way.
    // Records whose effect depends on unknown state are resolved exactly on the
    // book, positioned in the new timeline.
    void propagate(Sym& y, uint32_t i) {
        uint32_t end = y.vend;
        bool bnew = y.bvalid && y.pos <= i;
        for (uint32_t j = i + 1;; ++j) {
            if (delta.empty()) return;
            if (j % kCk == 0 && j <= end) {
                auto& cp = y.cpatch[j / kCk];
                cp.insert(cp.end(), delta.begin(), delta.end());
            }
            if (y.bvalid && !bnew && y.pos == j) {
                if (y.dict.nranked != y.dict.price.size()) y.dict.rerank();
                y.book.prepare(y.dict, y.rid, y.rtxt);
                for (const DEnt& e : delta) y.book.patch(e);
                bnew = true;
            }
            if (j >= end) return;
            if (j % kBlk == 0 && j + kBlk <= end && !touches(y, y.summary(j / kBlk))) {
                // Nothing in this block can see the difference: skip it whole.
                if (y.bvalid && !bnew && y.pos > j && y.pos < j + kBlk) {
                    if (y.dict.nranked != y.dict.price.size()) y.dict.rerank();
                    y.book.prepare(y.dict, y.rid, y.rtxt);
                    for (const DEnt& e : delta) y.book.patch(e);
                    bnew = true;
                }
                j += kBlk - 1;
                continue;
            }
            if (delta.size() > 64) {
                // Difference grew too large to track cheaply: replay from here instead.
                y.vend = j;
                if (y.bvalid && y.pos > j) y.bvalid = false;
                if (y.svalid && y.spos > j) y.svalid = false;
                return;
            }
            if (dstep(y, y.recs[j])) continue;
            if (!bnew) y.bvalid = false;
            y.svalid = false;
            seek(y, j);
            bnew = true;
            resolve(y, y.recs[j]);
            y.inval(j, j + 1);
            y.pos = j + 1;
        }
    }

    // Rewritten add with the same id and stock as the add it replaces.
    void rewrite_add(uint32_t idx, const Ev& e) {
        Ev& old = evs[idx];
        Sym& y = syms[old.rsym];
        size_t i = y.lower(e.seq);
        Rec orec = y.recs[i];
        old.side = e.side;
        old.shares = e.shares;
        old.price = e.price;
        Rec nrec = make_rec(old, y);
        y.recs[i] = nrec;
        y.inval(i, i + 1);
        if (y.svalid && y.spos > i) y.svalid = false;
        if (i >= y.vend) return;  // nothing validated past the rewrite
        if (orec.type != 'A' || orec.slot != nrec.slot) {
            y.dirty(static_cast<uint32_t>(i));
            return;
        }
        pend_delta.clear();
        pend_delta.push_back(DEnt{nrec.slot, DState{1, orec.side, orec.pid, orec.shares},
                                  DState{1, nrec.side, nrec.pid, nrec.shares}});
        if (pend_delta[0].o == pend_delta[0].n) return;
        pend_sym = old.rsym;
        pend_i = static_cast<uint32_t>(i);
    }

    // Completes deferred work; must run before anything reads or edits records.
    void settle() {
        if (pend_sym < 0) return;
        int32_t s = pend_sym;
        pend_sym = -1;
        delta.swap(pend_delta);
        propagate(syms[s], pend_i);
    }

    // Replaces (or inserts) the event at e.seq. New seqs are collected in `fresh`.
    void put(const Ev& e, std::vector<std::pair<uint32_t, uint32_t>>& fresh) {
        settle();
        uint32_t idx = find_seq(e.seq);
        if (idx != kNil) {
            const Ev& old = evs[idx];
            if (old.type == 'A' && e.type == 'A' && old.id == e.id && old.stock == e.stock && old.rsym >= 0) {
                rewrite_add(idx, e);
                return;
            }
            unlink_event(idx);
            evs[idx] = e;
        } else {
            idx = static_cast<uint32_t>(evs.size());
            evs.push_back(e);
            fresh.push_back({e.seq, idx});
        }
        if (e.seq == 0 && e.type != 'R' && e.type != 0) weird = true;
        link_event(idx);
        if (e.seq > max_seq) max_seq = e.seq;
    }

    void add_fresh(std::vector<std::pair<uint32_t, uint32_t>>& fresh) {
        if (fresh.empty()) return;
        std::sort(fresh.begin(), fresh.end());
        size_t n = gseq.size(), m = fresh.size();
        gseq.resize(n + m);
        gidx.resize(n + m);
        size_t a = n, b = m, w = n + m;
        while (b > 0) {
            if (a > 0 && gseq[a - 1] > fresh[b - 1].first) {
                --a;
                --w;
                gseq[w] = gseq[a];
                gidx[w] = gidx[a];
            } else {
                --b;
                --w;
                gseq[w] = fresh[b].first;
                gidx[w] = fresh[b].second;
            }
        }
    }

    void load_ck(Sym& y, uint32_t c, Book& b, uint32_t& pos, bool& valid) {
        b.load(y.cks[c]);
        for (const DEnt& e : y.cpatch[c]) b.patch(e);
        if (y.cpatch[c].size() > 32) {
            // Fold accumulated patches into the checkpoint itself.
            b.save(y.cks[c]);
            y.cpatch[c].clear();
        }
        pos = c * kCk;
        valid = true;
    }

    // Moves book b (holding the state after `pos` records when `valid`) to the
    // state after the first j records of y, extending the validated prefix.
    Book& seek_in(Sym& y, uint32_t j, Book& b, uint32_t& pos, bool& valid) {
        if (y.dict.nranked != y.dict.price.size()) y.dict.rerank();
        b.prepare(y.dict, y.rid, y.rtxt);
        uint32_t c = std::min(j, y.vend) / kCk;
        bool use_book = valid && pos <= j && pos + kCk / 2 >= c * kCk;
        if (!use_book) load_ck(y, c, b, pos, valid);
        Rec* r = y.recs.data();
        while (pos < j) {
            uint32_t next_ck = (pos / kCk + 1) * kCk;
            uint32_t stop = std::min(j, next_ck);
            b.foot_changed = false;
            for (uint32_t i = pos; i < stop; ++i) b.apply(r[i]);
            if (b.foot_changed) y.inval(pos, stop);
            pos = stop;
            if (stop == next_ck && stop > y.vend) {
                uint32_t m = stop / kCk;
                if (y.cks.size() <= m) {
                    y.cks.resize(m + 1);
                    y.cpatch.resize(m + 1);
                }
                b.save(y.cks[m]);
                y.cpatch[m].clear();
            }
            if (stop > y.vend) y.vend = stop;
        }
        return b;
    }

    // Book of symbol y after its first j records (the symbol's own book).
    Book& seek(Sym& y, uint32_t j) { return seek_in(y, j, y.book, y.pos, y.bvalid); }

    // Reads far from where the symbol's own book sits use its scratch book, so
    // the own book can stay at the tip (or the speculated position).
    Book& seek_read(int32_t s, uint32_t j) {
        Sym& y = syms[s];
        if (y.bvalid && y.pos <= j && j - y.pos <= kCk / 2) return seek(y, j);
        return seek_in(y, j, y.scratch, y.spos, y.svalid);
    }

    void format_at(int32_t s, uint32_t q) {
        settle();
        Sym& y = syms[s];
        uint32_t j = static_cast<uint32_t>(y.upper(q));
        Book* b;
        {
            b = &seek_read(s, j);
        }
        b->format(y.name);
    }
};

Fast* fast = nullptr;
ref::Engine* slow = nullptr;
std::vector<std::vector<uint8_t>> blobs;  // every APPLY payload (to rebuild the slow path)

void go_slow() {
    slow = new ref::Engine();
    for (auto& b : blobs) slow->apply_blob(b.data(), b.size());
    delete fast;
    fast = nullptr;
}

void do_apply(const uint8_t* p, size_t n) {
    // Parse first: a truncated frame aborts before any state change.
    static std::vector<Ev> batch;
    batch.clear();
    size_t i = 0, nf = 0;
    while (i + 6 <= n) {
        i += 6 + static_cast<size_t>((p[i + 4] << 8) | p[i + 5]);
        ++nf;
    }
    batch.reserve(nf);
    i = 0;
    while (i + 6 <= n) {
        uint32_t seq = be32(p + i);
        uint16_t len = static_cast<uint16_t>((p[i + 4] << 8) | p[i + 5]);
        i += 6;
        if (i + len > n) die(2);
        Ev e;
        if (decode(seq, p + i, len, e)) batch.push_back(e);
        i += len;
    }
    if (slow) {
        slow->apply_blob(p, n);
        return;
    }
    blobs.emplace_back(p, p + n);
    Fast& f = *fast;
    bool mono = true;
    if (f.ready)
        for (const Ev& e : batch)
            if (e.seq <= f.max_seq) {
                mono = false;
                break;
            }
    if (mono) {
        for (size_t k = 1; k < batch.size(); ++k)
            if (batch[k].seq <= batch[k - 1].seq) f.weird = true;
        if (!f.weird) {
            f.reserve_for(batch.size());
            for (const Ev& e : batch) f.append(e);
        }
    } else {
        // Later frames for the same seq win; order otherwise irrelevant.
        std::stable_sort(batch.begin(), batch.end(), [](const Ev& a, const Ev& b) { return a.seq < b.seq; });
        size_t w = 0;
        for (size_t k = 0; k < batch.size(); ++k) {
            if (k + 1 < batch.size() && batch[k + 1].seq == batch[k].seq) continue;
            batch[w++] = batch[k];
        }
        batch.resize(w);
        static std::vector<std::pair<uint32_t, uint32_t>> fresh;
        fresh.clear();
        f.reserve_for(batch.size());
        for (const Ev& e : batch) f.put(e, fresh);
        f.add_fresh(fresh);
    }
    f.merge_pending();
    if (mono && !f.weird && batch.size() > 4096) {
        // Large load: bring every symbol's checkpoints up to the tip before
        // replying (small appends are extended while idle instead).
        for (auto& y : f.syms)
            if (y.vend < y.recs.size()) f.seek(y, static_cast<uint32_t>(y.recs.size()));
    }
    f.ready = true;
    if (f.has_conflict)
        for (const Sym& y : f.syms)
            if (y.recs.size() >= 13000) f.weird = true;  // reference would snapshot it
    if (f.weird) {
        go_slow();
        return;
    }
}

void do_query(const std::vector<std::pair<std::string, uint32_t>>& qs) {
    if (fast && fast->has_conflict)
        for (const auto& q : qs)
            if (q.second >= fast->max_seq) {
                go_slow();  // tip answers come from the reference's live books
                break;
            }
    if (slow) {
        std::string s;
        slow->query(qs, s);
        out.put(s);
        return;
    }
    Fast& f = *fast;
    for (const auto& [name, seq] : qs) {
        int32_t s = -1;
        if (name.size() <= 8) {
            uint64_t k;
            char nm[8];
            memset(nm, ' ', 8);
            memcpy(nm, name.data(), name.size());
            memcpy(&k, nm, 8);
            uint32_t v = f.sym_of.find(k);
            if (v != U64Map::kNone) s = static_cast<int32_t>(v);
        }
        if (s < 0) {
            out.put(name);
            out.put(" B - A -\n", 9);
            continue;
        }
        f.format_at(s, seq);
        out.put("\n", 1);
    }
    out.put("END\nREADY\n", 10);
}

void do_snap(uint32_t seq) {
    if (fast && fast->has_conflict && seq >= fast->max_seq) go_slow();
    if (slow) {
        std::string s;
        slow->snap(seq, s);
        out.put(s);
        return;
    }
    Fast& f = *fast;
    std::vector<std::pair<size_t, size_t>> lines;
    size_t base = out.n;
    for (int32_t s = 0; s < static_cast<int32_t>(f.syms.size()); ++s) {
        size_t st = out.n;
        f.format_at(s, seq);
        size_t len = out.n - st;
        static const char kEmpty[] = " B - A -";
        if (memmem(out.p + st, len, kEmpty, 8)) {
            out.n = st;
            continue;
        }
        lines.push_back({st - base, len});
    }
    std::string tmp(out.p + base, out.n - base);
    std::sort(lines.begin(), lines.end(), [&](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
        int c = memcmp(tmp.data() + a.first, tmp.data() + b.first, std::min(a.second, b.second));
        if (c != 0) return c < 0;
        return a.second < b.second;
    });
    out.n = base;
    for (auto& [off, len] : lines) {
        out.put(tmp.data() + off, len);
        out.put("\n", 1);
    }
    out.put("END\nREADY\n", 10);
}

}  // namespace

int main() {
    mallopt(M_MMAP_THRESHOLD, 1 << 30);
    mallopt(M_TRIM_THRESHOLD, 1 << 30);
    in.init();
    fast = new Fast();
    std::vector<std::pair<std::string, uint32_t>> qs;
    while (true) {
        auto [s, n] = in.line();
        if (n >= 6 && memcmp(s, "APPLY ", 6) == 0) {
            uint64_t len = parse_u64(s + 6, n - 6);
            if (len > (1ULL << 40)) die_abort();
            const uint8_t* p = in.bytes(len);
            do_apply(p, len);
            out.put("READY\n", 6);
        } else if (n >= 6 && memcmp(s, "QUERY ", 6) == 0) {
            int k = parse_int(s + 6, n - 6);
            qs.clear();
            for (int i = 0; i < k; ++i) {
                auto [r, rn] = in.line();
                const char* sp = static_cast<const char*>(memchr(r, ' ', rn));
                std::string name;
                uint64_t v;
                if (sp) {
                    name.assign(r, sp);
                    v = parse_u64(sp + 1, rn - static_cast<size_t>(sp + 1 - r));
                } else {
                    name.assign(r, rn);
                    v = parse_u64(r, rn);
                }
                qs.push_back({name, static_cast<uint32_t>(v)});
            }
            do_query(qs);
        } else if (n >= 5 && memcmp(s, "SNAP ", 5) == 0) {
            uint64_t v = parse_u64(s + 5, n - 5);
            do_snap(static_cast<uint32_t>(v));
        } else if (n == 4 && memcmp(s, "QUIT", 4) == 0) {
            out.flush();
            _exit(0);
        } else {
            die(2);
        }
    }
}
