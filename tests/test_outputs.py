#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""判分输出：10 条行为测试，按重要程度加权，权重和 = 1。

每条测试自己造一份台架试验记录，并用**另一套独立实现**（坐标上升 + 黄金分割）
复算三参数 Weibull 的极大似然解作为期望值，再驱动被测程序的命令行入口比对报告。
期望值不读被测代码，也不读参考解的输出。
"""
import json
import math
import os
import subprocess
import sys
import tempfile

APP = os.environ.get("CT_APP_DIR", "/app")
ENTRY = os.path.join(APP, "src", "reliability.py")

WEIGHTS = {
    "test_params_match_independent_mle_on_base_dataset": 0.16,
    "test_params_match_second_dataset_with_different_parameters": 0.13,
    "test_censoring_handled_under_heavy_censoring": 0.12,
    "test_b_life_matches_independent_computation": 0.10,
    "test_two_parameter_limit_when_gamma_near_zero": 0.08,
    "test_invalid_records_rejected_without_output": 0.12,
    "test_report_schema_and_rounding": 0.08,
    "test_rerun_produces_identical_bytes": 0.06,
    "test_failed_run_keeps_previous_report": 0.06,
    "test_large_sample_recovers_parameters": 0.09,
}


# ---------------------------------------------------------------- 造数据（独立实现）
class _Lcg(object):
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next_u(self):
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return (self.state % 1000003) / 1000003.0


def make_tests(beta, eta, gamma, n=48, seed=7, censor_time=None, censor_every=3, pct=10):
    rng = _Lcg(seed)
    records = []
    for i in range(n):
        u = rng.next_u()
        t = gamma + eta * (-math.log(1.0 - u * 0.999999)) ** (1.0 / beta)
        failed = True
        if censor_time is not None and censor_every and i % censor_every == censor_every - 1 and t > censor_time:
            t = censor_time
            failed = False
        records.append({"id": "u%03d" % (i + 1), "time": round(t, 1), "failed": failed})
    return {"unit": "kcycles", "records": records, "reporting": {"b_life_percent": pct}}


# ---------------------------------------------------------------- 独立 MLE（坐标上升）
def _eta_closed(beta, gamma, times, failed):
    r = 0
    s = 0.0
    for t, f in zip(times, failed):
        z = t - gamma
        if z <= 0:
            return None
        if f:
            r += 1
        s += z ** beta
    return (s / r) ** (1.0 / beta), r


def _lnl(beta, gamma, times, failed):
    got = _eta_closed(beta, gamma, times, failed)
    if got is None:
        return None
    eta, r = got
    acc = r * math.log(beta) - r * beta * math.log(eta)
    for t, f in zip(times, failed):
        if f:
            acc += (beta - 1.0) * math.log(t - gamma) - ((t - gamma) / eta) ** beta
        else:
            acc -= ((t - gamma) / eta) ** beta
    return acc


def _golden_max(fn, lo, hi, iters=70):
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(iters):
        if fc is None or fd is None:
            break
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = fn(d)
    return (a + b) / 2.0


def mle_solution(times, failed, pct):
    """粗网格定位全局峰，再在 (γ, β) 上黄金分割细化，避免坐标上升掉进局部极值。"""
    min_fail = min(t for t, f in zip(times, failed) if f)
    g_hi = min_fail * 0.995
    best = None
    for gi in range(61):
        gamma = g_hi * gi / 60.0
        for bi in range(41):
            beta = math.exp(math.log(0.08) + (math.log(15.0) - math.log(0.08)) * bi / 40.0)
            val = _lnl(beta, gamma, times, failed)
            if val is not None and (best is None or val > best[0]):
                best = (val, beta, gamma)
    if best is None:
        raise RuntimeError("没有可行解")
    beta, gamma = best[1], best[2]
    for _ in range(6):
        log_beta = _golden_max(lambda x: _lnl(math.exp(x), gamma, times, failed),
                               math.log(max(0.08, beta / 1.6)), math.log(min(15.0, beta * 1.6)))
        beta = math.exp(log_beta)
        gamma = _golden_max(lambda g: _lnl(beta, g, times, failed),
                            max(0.0, gamma - g_hi / 30.0), min(g_hi, gamma + g_hi / 30.0))
        if gamma < 1e-6:
            gamma = 0.0
    ln_l = _lnl(beta, gamma, times, failed)
    eta, _r = _eta_closed(beta, gamma, times, failed)
    b_life = gamma + eta * (-math.log(1.0 - pct / 100.0)) ** (1.0 / beta)
    return {"beta": beta, "eta": eta, "gamma": gamma, "loglik": ln_l, "b_life": b_life}


# ---------------------------------------------------------------- 驱动被测程序
def _run(tests_path, out_path, timeout=240):
    cmd = [sys.executable, ENTRY, "--tests", tests_path, "--out", out_path]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def run_case(data):
    tmp = tempfile.mkdtemp(prefix="wb-case-")
    tpath = os.path.join(tmp, "tests.json")
    opath = os.path.join(tmp, "report.json")
    with open(tpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    proc = _run(tpath, opath)
    report = None
    if os.path.exists(opath):
        try:
            with open(opath, encoding="utf-8") as f:
                report = json.load(f)
        except ValueError:
            report = None
    return report, {"tests": tpath, "out": opath, "proc": proc, "dir": tmp}


def _expect(data):
    times = [float(r["time"]) for r in data["records"]]
    failed = [bool(r["failed"]) for r in data["records"]]
    return mle_solution(times, failed, float(data["reporting"]["b_life_percent"]))


# 位置参数 γ 是三参数 Weibull 里最难定的一位（似然面在边界附近很平），
# 所以 β/η 卡 0.5%，γ 卡 3% 或 1 个时间单位（取大者）。
_PARAM_TOL = {"beta": 0.005, "eta": 0.005, "gamma": 0.03}


def _params_close(report, exp):
    if not isinstance(report, dict):
        return False
    p = report.get("params") or {}
    for key in ("beta", "eta", "gamma"):
        got = float(p.get(key))
        if not math.isfinite(got):
            return False
        base = exp[key]
        limit = _PARAM_TOL[key]
        if abs(base) < 1e-6:
            if abs(got) > 1.0:
                return False
        elif abs(got - base) > max(limit * abs(base), 1.0 if key == "gamma" else 0.0):
            return False
    return True


# ---------------------------------------------------------------- 测试
def test_params_match_independent_mle_on_base_dataset():
    """题面原文：「按极大似然拟合三参数 Weibull」，失效件用 ln f(t)、未失效件只用 ln S(t)。

    随机台架记录（约两成右删失）：三个参数与独立复算的极大似然解相对误差不超过 0.5%。
    """
    data = make_tests(2.35, 820.0, 95.0, n=48, seed=11, censor_time=700.0)
    report, _ = run_case(data)
    return _params_close(report, _expect(data))


def test_params_match_second_dataset_with_different_parameters():
    """题面原文：「复检时程序会被喂别的批次，样本数、删失比例与参数都和前一批不同」。

    换一组参数（β≈1.4、η≈1500、γ≈300）与更重的删失后仍要吻合。
    """
    data = make_tests(1.4, 1500.0, 300.0, n=40, seed=23, censor_time=1100.0, censor_every=2)
    report, _ = run_case(data)
    return _params_close(report, _expect(data))


def test_censoring_handled_under_heavy_censoring():
    """题面原文：「未失效的样件只知道撑过停机时刻」。

    一半以上样件右删失时，把删失件当失效件处理会明显高估尺度参数，本用例卡住这种写法。
    """
    data = make_tests(2.0, 900.0, 50.0, n=60, seed=31, censor_time=600.0, censor_every=1)
    report, _ = run_case(data)
    return _params_close(report, _expect(data))


def test_b_life_matches_independent_computation():
    """题面原文：「报告给出 reporting.b_life_percent 对应的 B 寿命」。

    B_p = γ + η·(-ln(1-p/100))^(1/β)，取 p=5% 的批次核对，相对误差不超过 0.5%。
    """
    data = make_tests(1.8, 1200.0, 0.0, n=44, seed=47, censor_time=900.0, censor_every=3, pct=5)
    report, _ = run_case(data)
    if not isinstance(report, dict):
        return False
    exp = _expect(data)
    try:
        got = float(report["b_life"])
    except (KeyError, TypeError, ValueError):
        return False
    return abs(got - exp["b_life"]) <= 0.005 * abs(exp["b_life"])


def test_two_parameter_limit_when_gamma_near_zero():
    """题面原文：「位置参数估出负数或接近零时按零处理，退化成两参数」。

    真值为两参数（γ=0）的批次，估出的 γ 不能凭空顶到几十个 kcycles。
    """
    data = make_tests(2.6, 700.0, 0.0, n=50, seed=53, censor_time=None)
    report, _ = run_case(data)
    if not _params_close(report, _expect(data)):
        return False
    return float(report["params"]["gamma"]) <= 0.15 * float(report["params"]["eta"])


def test_invalid_records_rejected_without_output():
    """题面原文：「记录不合法时以退出码 2 结束，不产出报告」。

    四种坏输入：全部删失、id 重复、时间为负、记录不足 4 条 —— 都要退出码 2、不落文件、stderr 非空。
    """
    good = make_tests(2.0, 800.0, 60.0, n=10, seed=61, censor_time=500.0)
    all_cens = json.loads(json.dumps(good))
    for rec in all_cens["records"]:
        rec["failed"] = False
    dup = json.loads(json.dumps(good))
    dup["records"][1]["id"] = dup["records"][0]["id"]
    neg = json.loads(json.dumps(good))
    neg["records"][2]["time"] = -5.0
    few = json.loads(json.dumps(good))
    few["records"] = few["records"][:3]
    for data in (all_cens, dup, neg, few):
        tmp = tempfile.mkdtemp(prefix="wb-bad-")
        tpath = os.path.join(tmp, "tests.json")
        opath = os.path.join(tmp, "report.json")
        with open(tpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        proc = _run(tpath, opath)
        if proc.returncode != 2 or os.path.exists(opath):
            return False
        if not (proc.stderr or "").strip():
            return False
        if [n for n in os.listdir(tmp) if n.endswith(".tmp")]:
            return False
    return True


def test_report_schema_and_rounding():
    """题面原文：「报告结构固定，参数保留三位小数、寿命保留一位小数」。

    校验顶层字段、计数、参数位数、数值有限性，以及文件里不出现 NaN 或 Infinity。
    """
    data = make_tests(2.2, 850.0, 80.0, n=36, seed=67, censor_time=700.0)
    report, ctx = run_case(data)
    if not isinstance(report, dict):
        return False
    for key in ("n", "failures", "censored", "params", "b_life", "b_life_percent", "loglik", "meta"):
        if key not in report:
            return False
    if report["n"] != len(data["records"]):
        return False
    fails = sum(1 for r in data["records"] if r["failed"])
    if report["failures"] != fails or report["censored"] != len(data["records"]) - fails:
        return False
    for key in ("beta", "eta", "gamma"):
        v = float(report["params"][key])
        if not math.isfinite(v) or abs(v - round(v, 3)) > 1e-12:
            return False
    if abs(float(report["b_life"]) - round(float(report["b_life"]), 1)) > 1e-12:
        return False
    raw = open(ctx["out"], encoding="utf-8").read()
    return "NaN" not in raw and "Infinity" not in raw


def test_rerun_produces_identical_bytes():
    """题面原文：「同一份试验记录重复运行，报告逐字节一致」。

    连跑两次，两个报告文件字节完全相同。
    """
    data = make_tests(2.35, 820.0, 95.0, n=48, seed=11, censor_time=700.0)
    first, ctx1 = run_case(data)
    if first is None:
        return False
    body1 = open(ctx1["out"], "rb").read()
    second, ctx2 = run_case(data)
    if second is None:
        return False
    return body1 == open(ctx2["out"], "rb").read()


def test_failed_run_keeps_previous_report():
    """题面原文：「运行失败时不能覆盖上一次已经写好的报告」。

    先跑一次合格记录留下报告，再用重复 id 的坏记录跑：退出码 2，旧报告字节不变。
    """
    good = make_tests(2.0, 900.0, 70.0, n=30, seed=71, censor_time=600.0)
    tmp = tempfile.mkdtemp(prefix="wb-keep-")
    tpath = os.path.join(tmp, "tests.json")
    opath = os.path.join(tmp, "report.json")
    with open(tpath, "w", encoding="utf-8") as f:
        json.dump(good, f, ensure_ascii=False)
    first = _run(tpath, opath)
    if first.returncode != 0 or not os.path.exists(opath):
        return False
    before = open(opath, "rb").read()
    broken = json.loads(json.dumps(good))
    broken["records"][1]["id"] = broken["records"][0]["id"]
    with open(tpath, "w", encoding="utf-8") as f:
        json.dump(broken, f, ensure_ascii=False)
    second = _run(tpath, opath)
    if second.returncode != 2:
        return False
    return open(opath, "rb").read() == before


def test_large_sample_recovers_parameters():
    """题面原文：「样本量变大时参数应回到真值附近」。

    300 件的批次（约三成删失）：形状与尺度与真值相差不超过 10% / 6%，位置参数落在真值的一半到一倍半之间。
    """
    truth = {"beta": 2.0, "eta": 1000.0, "gamma": 150.0}
    data = make_tests(truth["beta"], truth["eta"], truth["gamma"], n=300, seed=83,
                      censor_time=800.0, censor_every=3)
    report, _ = run_case(data)
    if not isinstance(report, dict):
        return False
    p = report.get("params") or {}
    try:
        beta, eta, gamma = float(p["beta"]), float(p["eta"]), float(p["gamma"])
    except (KeyError, TypeError, ValueError):
        return False
    return (abs(beta - truth["beta"]) <= 0.10 * truth["beta"]
            and abs(eta - truth["eta"]) <= 0.06 * truth["eta"]
            and 0.5 * truth["gamma"] <= gamma <= 1.5 * truth["gamma"])


def main():
    total = 0.0
    for name, weight in WEIGHTS.items():
        fn = globals().get(name)
        try:
            passed = bool(fn and fn())
        except Exception as e:  # noqa: BLE001 —— 任何异常都按不通过计
            print("%s: 异常 %r" % (name, e), file=sys.stderr)
            passed = False
        total += weight * passed
        print("%s: %s (权重 %.2f)" % (name, "PASS" if passed else "FAIL", weight), file=sys.stderr)
    print("%.6f" % total)


if __name__ == "__main__":
    main()
