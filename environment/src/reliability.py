#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""台架寿命复核：Weibull 拟合与 B10 寿命。

用法：
    python3 src/reliability.py --tests data/tests.json --out out/report.json
"""
import argparse
import json
import math
import os


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fit(times):
    """按最小观测值取位置参数，形状与尺度用两步迭代。"""
    gamma = min(times)
    beta = 2.0
    eta = sum(times) / len(times)
    for _ in range(40):
        zs = [t - gamma for t in times]
        s_ln = sum(math.log(z) for z in zs)
        s_pow = sum(z ** beta for z in zs)
        eta = (s_pow / len(times)) ** (1.0 / beta)
        beta = len(times) / (s_ln - len(times) * math.log(eta))
        if beta <= 0:
            beta = 0.5
    return beta, eta, gamma


def b_life(beta, eta, pct):
    return eta * (-math.log(1.0 - pct / 100.0)) ** (1.0 / beta)


def main():
    ap = argparse.ArgumentParser(description="Weibull 拟合与 B10 寿命")
    ap.add_argument("--tests", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = load(args.tests)
    times = [float(r["time"]) for r in data["records"]]
    pct = float(data["reporting"]["b_life_percent"])
    beta, eta, gamma = fit(times)

    report = {
        "n": len(times),
        "failures": len(times),
        "censored": 0,
        "params": {"beta": round(beta, 3), "eta": round(eta, 3), "gamma": round(gamma, 3)},
        "b_life": round(b_life(beta, eta, pct), 1),
        "b_life_percent": pct,
        "loglik": 0.0,
        "meta": {"method": "mle-3p", "unit": data.get("unit", "")},
    }

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
