#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成轴承台架寿命试验记录（三参数 Weibull，含右删失）。

数据口径：业务模型 + 确定性生成规则（固定伪随机序列，不用真随机）——
  t = γ + η·(-ln(1-u))^(1/β)，u 由固定种子 LCG 产生；
  抽中的样件到 censor_time 还没坏就停机、按右删失记录（failed=false，time 记停机时刻）。
  时间保留一位小数（台架计时精度），整份数据可逐字节复现。
"""
import json
import math
import sys


class Lcg(object):
    """固定种子的线性同余，保证数据可复现。"""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next_u(self):
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return (self.state % 1000003) / 1000003.0


def sample_times(beta, eta, gamma, n, seed, censor_time, censor_every):
    rng = Lcg(seed)
    records = []
    for i in range(n):
        u = rng.next_u()
        t = gamma + eta * (-math.log(1.0 - u * 0.999999)) ** (1.0 / beta)
        failed = True
        if censor_every and i % censor_every == censor_every - 1 and t > censor_time:
            t = censor_time
            failed = False
        records.append({"id": "s%02d" % (i + 1), "time": round(t, 1), "failed": failed})
    return records


def build(beta=2.35, eta=820.0, gamma=95.0, n=48, seed=20261022,
          censor_time=700.0, censor_every=3, b_life_percent=10):
    return {
        "unit": "kcycles",
        "records": sample_times(beta, eta, gamma, n, seed, censor_time, censor_every),
        "reporting": {"b_life_percent": b_life_percent},
    }


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "tests.json"
    data = build()
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    n = len(data["records"])
    print("wrote %s  n=%d  failures=%d  censored=%d"
          % (out, n, sum(1 for r in data["records"] if r["failed"]),
             sum(1 for r in data["records"] if not r["failed"])))
