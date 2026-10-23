#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""台架寿命复核：三参数 Weibull 极大似然拟合（含右删失）与 B10 寿命。

命令行：
    python3 src/reliability.py --tests data/tests.json --out out/report.json

口径：
  · 失效件对似然的贡献是 ln f(t) = ln(β/η) + (β-1)·ln z - z^β，z=(t-γ)/η；
  · 右删失件只贡献 ln S(t) = -z^β（不知道确切寿命，只知道「撑过停机时刻」）；
  · 给定 β、γ 时 η 有闭式解 η^β = Σ_all z^β / r（r 是失效件数），
    所以只需在 (γ, β) 上做二维搜索：γ 用粗网格 + 黄金分割细化，β 用黄金分割；
  · 位置参数 γ 不得超过最小失效时间，估出负数时按 0 处理（退化成两参数）；
  · 报告里的 B_p = γ + η·(-ln(1-p/100))^(1/β)；
  · 输出先写临时文件再原子替换；非法输入以退出码 2 结束且不落文件。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

EXIT_INVALID = 2


def _fail(msg):
    sys.stderr.write("非法输入：%s\n" % msg)
    raise SystemExit(EXIT_INVALID)


def load_tests(path):
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        _fail("读不到试验记录 %s" % path)
    except ValueError as e:
        _fail("试验记录不是合法 JSON：%s" % e)
    if not isinstance(raw, dict):
        _fail("试验记录顶层要是对象")
    records = raw.get("records")
    if not isinstance(records, list) or len(records) < 4:
        _fail("records 至少要 4 条")
    reporting = raw.get("reporting")
    if not isinstance(reporting, dict):
        _fail("缺 reporting")
    pct = reporting.get("b_life_percent")
    if not isinstance(pct, (int, float)) or isinstance(pct, bool) or not (0 < float(pct) < 100):
        _fail("reporting.b_life_percent 要在 0 与 100 之间")

    ids = set()
    times, failed = [], []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            _fail("records[%d] 不是对象" % i)
        rid = rec.get("id")
        if not isinstance(rid, str) or not rid.strip():
            _fail("records[%d].id 不合法" % i)
        if rid in ids:
            _fail("records[%d].id 重复：%s" % (i, rid))
        ids.add(rid)
        t = rec.get("time")
        if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(float(t)) or float(t) <= 0:
            _fail("records[%d].time 要是正数" % i)
        f = rec.get("failed")
        if not isinstance(f, bool):
            _fail("records[%d].failed 要是布尔值" % i)
        times.append(float(t))
        failed.append(f)

    n_fail = sum(1 for v in failed if v)
    if n_fail < 3:
        _fail("失效件只有 %d 个，三参数拟合至少要 3 个" % n_fail)
    return {
        "times": times,
        "failed": failed,
        "n": len(times),
        "n_fail": n_fail,
        "pct": float(pct),
        "unit": str(raw.get("unit") or ""),
    }


# ---------------------------------------------------------------- 似然
def _profile(beta, gamma, times, failed):
    """给定 β、γ，用闭式解出 η，返回 (η, lnL)；有 z<=0 时返回 None。"""
    zs = []
    for t in times:
        z = t - gamma
        if z <= 0.0:
            return None
        zs.append(z)
    r = sum(1 for f in failed if f)
    s_all = 0.0
    for z in zs:
        s_all += z ** beta
    eta = (s_all / r) ** (1.0 / beta)
    sum_ln_fail = 0.0
    for z, f in zip(zs, failed):
        if f:
            sum_ln_fail += math.log(z)
    ln_l = r * math.log(beta) - r * beta * math.log(eta) + (beta - 1.0) * sum_ln_fail - r
    return eta, ln_l


def _best_beta(gamma, times, failed, lo=0.05, hi=20.0, iters=120):
    """黄金分割搜 β（对数尺度，避免小 β 区被压扁）。"""
    a, b = math.log(lo), math.log(hi)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = _profile(math.exp(c), gamma, times, failed)
    fd = _profile(math.exp(d), gamma, times, failed)
    for _ in range(iters):
        if fc is None or fd is None:
            return None
        if not (fc[1] < fd[1]):
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = _profile(math.exp(c), gamma, times, failed)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = _profile(math.exp(d), gamma, times, failed)
    beta = math.exp((a + b) / 2.0)
    res = _profile(beta, gamma, times, failed)
    if res is None:
        return None
    return beta, res[0], res[1]


def fit(scan):
    times, failed = scan["times"], scan["failed"]
    min_fail = min(t for t, f in zip(times, failed) if f)
    g_max = min_fail * 0.995

    # 1) γ 粗网格
    steps = 200
    best = None
    for i in range(steps + 1):
        gamma = g_max * i / float(steps)
        got = _best_beta(gamma, times, failed)
        if got and (best is None or got[2] > best[3]):
            best = (gamma, got[0], got[1], got[2])

    # 2) γ 黄金分割细化
    if g_max > 0:
        lo = max(0.0, best[0] - g_max / steps)
        hi = min(g_max, best[0] + g_max / steps)
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        for _ in range(60):
            c = hi - phi * (hi - lo)
            d = lo + phi * (hi - lo)
            gc = _best_beta(c, times, failed)
            gd = _best_beta(d, times, failed)
            if gc and (not gd or not (gc[2] < gd[2])):
                hi = d
            else:
                lo = c
            got = gc if gc and (not gd or not (gc[2] < gd[2])) else gd
            if got and got[2] > best[3]:
                best = (c if got is gc else d, got[0], got[1], got[2])

    gamma, beta, eta, ln_l = best
    if gamma < 1e-6:
        zero = _best_beta(0.0, times, failed)
        if zero and not (zero[2] < ln_l):
            gamma, beta, eta, ln_l = 0.0, zero[0], zero[1], zero[2]
    return beta, eta, gamma, ln_l


def b_life(beta, eta, gamma, pct):
    return gamma + eta * (-math.log(1.0 - pct / 100.0)) ** (1.0 / beta)


def build_report(scan, beta, eta, gamma, ln_l):
    return {
        "n": scan["n"],
        "failures": scan["n_fail"],
        "censored": scan["n"] - scan["n_fail"],
        "params": {"beta": round(beta, 3), "eta": round(eta, 3), "gamma": round(gamma, 3)},
        "b_life": round(b_life(beta, eta, gamma, scan["pct"]), 1),
        "b_life_percent": scan["pct"],
        "loglik": round(ln_l, 6),
        "meta": {"method": "mle-3p", "unit": scan["unit"]},
    }


def atomic_write(path, obj):
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    body = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=".rel-", suffix=".tmp", dir=out_dir or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description="三参数 Weibull 拟合与 B10 寿命")
    ap.add_argument("--tests", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    scan = load_tests(args.tests)
    beta, eta, gamma, ln_l = fit(scan)
    atomic_write(args.out, build_report(scan, beta, eta, gamma, ln_l))
    return 0


if __name__ == "__main__":
    sys.exit(main())
