# -*- coding: utf-8 -*-
"""sxw-003 判分测试（pytest 版）：每条测试自己造台架记录，断言写在测试函数体内。

期望值由本文件独立实现的三参数 Weibull 极大似然（粗网格 + 黄金分割细化）按当前数据现算，
不读被测代码，也不与参考解的输出比对。权重见 conftest.py 的 WEIGHTS。
"""
import json
import math
import os
import subprocess
import sys
import tempfile

APP = os.environ.get("CT_APP_DIR", "/app")
ENTRY = os.path.join(APP, "src", "reliability.py")


# ---------------------------------------------------------------- 造记录（独立实现）
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
    assert best is not None, "独立复算没有可行解"
    beta, gamma = best[1], best[2]
    for _ in range(6):
        log_beta = _golden_max(lambda x: _lnl(math.exp(x), gamma, times, failed),
                               math.log(max(0.08, beta / 1.6)), math.log(min(15.0, beta * 1.6)))
        beta = math.exp(log_beta)
        gamma = _golden_max(lambda g: _lnl(beta, g, times, failed),
                            max(0.0, gamma - g_hi / 30.0), min(g_hi, gamma + g_hi / 30.0))
        if gamma < 1e-6:
            gamma = 0.0
    eta, _r = _eta_closed(beta, gamma, times, failed)
    return {"beta": beta, "eta": eta, "gamma": gamma,
            "b_life": gamma + eta * (-math.log(1.0 - pct / 100.0)) ** (1.0 / beta)}


def ln_likelihood(beta, eta, gamma, data):
    """三参数 Weibull 的对数似然：失效件用密度项，未失效件只用生存项。"""
    acc = 0.0
    for s in data["records"]:
        z = float(s["time"]) - gamma
        if z <= 0:
            return float("-inf")
        t = z / eta
        if s.get("failed"):
            acc += math.log(beta) - math.log(eta) + (beta - 1.0) * math.log(t) - t ** beta
        else:
            acc += -(t ** beta)
    return acc


def expect(data):
    times = [float(r["time"]) for r in data["records"]]
    failed = [bool(r["failed"]) for r in data["records"]]
    return mle_solution(times, failed, float(data["reporting"]["b_life_percent"]))


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


# ---------------------------------------------------------------- 测试
def test_params_match_independent_mle_on_base_dataset():
    """题面原文：「按极大似然拟合三参数 Weibull」，失效件用密度项、未失效件只用生存项。

    随机台架记录（约两成右删失）：β/η 相对误差不超过 0.5%，γ 不超过 3% 或 1 个单位。
    """
    data = make_tests(2.35, 820.0, 95.0, n=48, seed=11, censor_time=700.0)
    report, _ = run_case(data)
    assert isinstance(report, dict), "没有产出报告"
    exp = expect(data)
    params = report.get("params") or {}
    beta = float(params.get("beta", float("nan")))
    eta = float(params.get("eta", float("nan")))
    gamma = float(params.get("gamma", float("nan")))
    assert math.isfinite(beta), "β 不是有限值：%r" % beta
    assert math.isfinite(eta), "η 不是有限值：%r" % eta
    assert math.isfinite(gamma), "γ 不是有限值：%r" % gamma
    assert abs(beta - exp["beta"]) <= 0.0025 * abs(exp["beta"]), \
        "β 偏差 %.4f 超容差（基准 %.4f）" % (beta - exp["beta"], exp["beta"])
    assert abs(eta - exp["eta"]) <= 0.0025 * abs(exp["eta"]), \
        "η 偏差 %.4f 超容差（基准 %.4f）" % (eta - exp["eta"], exp["eta"])
    assert abs(gamma - exp["gamma"]) <= max(0.02 * abs(exp["gamma"]), 1.0), \
        "γ 偏差 %.4f 超容差（基准 %.4f）" % (gamma - exp["gamma"], exp["gamma"])



def test_params_match_second_dataset_with_different_parameters():
    """题面原文：「下一批复检的样本数、删失比例和参数都会换」。

    换 β≈1.4、η≈1500、γ≈300 与更重删失的批次后仍要吻合。
    """
    data = make_tests(1.4, 1500.0, 300.0, n=40, seed=23, censor_time=1100.0, censor_every=2)
    report, _ = run_case(data)
    assert isinstance(report, dict), "没有产出报告"
    exp = expect(data)
    params = report.get("params") or {}
    beta = float(params.get("beta", float("nan")))
    eta = float(params.get("eta", float("nan")))
    gamma = float(params.get("gamma", float("nan")))
    assert math.isfinite(beta), "β 不是有限值：%r" % beta
    assert math.isfinite(eta), "η 不是有限值：%r" % eta
    assert math.isfinite(gamma), "γ 不是有限值：%r" % gamma
    assert abs(beta - exp["beta"]) <= 0.0025 * abs(exp["beta"]), \
        "β 偏差 %.4f 超容差（基准 %.4f）" % (beta - exp["beta"], exp["beta"])
    assert abs(eta - exp["eta"]) <= 0.0025 * abs(exp["eta"]), \
        "η 偏差 %.4f 超容差（基准 %.4f）" % (eta - exp["eta"], exp["eta"])
    assert abs(gamma - exp["gamma"]) <= max(0.02 * abs(exp["gamma"]), 1.0), \
        "γ 偏差 %.4f 超容差（基准 %.4f）" % (gamma - exp["gamma"], exp["gamma"])



def test_censoring_handled_under_heavy_censoring():
    """题面原文：「未失效件只能用生存函数项」。

    一半以上样件右删失时，把删失件当失效件处理会明显高估尺度参数，本用例卡住这种写法。
    """
    data = make_tests(2.0, 900.0, 50.0, n=60, seed=31, censor_time=600.0, censor_every=1)
    report, _ = run_case(data)
    assert isinstance(report, dict), "没有产出报告"
    exp = expect(data)
    params = report.get("params") or {}
    beta = float(params.get("beta", float("nan")))
    eta = float(params.get("eta", float("nan")))
    gamma = float(params.get("gamma", float("nan")))
    assert math.isfinite(beta), "β 不是有限值：%r" % beta
    assert math.isfinite(eta), "η 不是有限值：%r" % eta
    assert math.isfinite(gamma), "γ 不是有限值：%r" % gamma
    assert abs(beta - exp["beta"]) <= 0.0025 * abs(exp["beta"]), \
        "β 偏差 %.4f 超容差（基准 %.4f）" % (beta - exp["beta"], exp["beta"])
    assert abs(eta - exp["eta"]) <= 0.0025 * abs(exp["eta"]), \
        "η 偏差 %.4f 超容差（基准 %.4f）" % (eta - exp["eta"], exp["eta"])
    assert abs(gamma - exp["gamma"]) <= max(0.02 * abs(exp["gamma"]), 1.0), \
        "γ 偏差 %.4f 超容差（基准 %.4f）" % (gamma - exp["gamma"], exp["gamma"])



def test_b_life_matches_independent_computation():
    """题面原文：「指定的那个失效百分位对应的寿命」。

    B_p = γ + η·(-ln(1-p/100))^(1/β)，取 p=5% 的批次核对，相对误差不超过 0.5%。
    """
    data = make_tests(1.8, 1200.0, 0.0, n=44, seed=47, censor_time=900.0, censor_every=3, pct=5)
    report, _ = run_case(data)
    assert isinstance(report, dict), "没有产出报告"
    exp = expect(data)
    assert "b_life" in report, "报告缺 b_life"
    got = float(report["b_life"])
    assert abs(got - exp["b_life"]) <= 0.003 * abs(exp["b_life"]), "B 寿命偏差 %.2f%%" % (
        (got - exp["b_life"]) / exp["b_life"] * 100)


def test_two_parameter_limit_when_gamma_near_zero():
    """题面原文：「估出的位置参数为负或接近零时按 0 处理」。

    真值为两参数的批次，估出的 γ 不能凭空顶到几十个 kcycles。
    """
    data = make_tests(2.6, 700.0, 0.0, n=50, seed=53, censor_time=None)
    report, _ = run_case(data)
    assert isinstance(report, dict), "没有产出报告"
    exp = expect(data)
    params = report.get("params") or {}
    beta = float(params.get("beta", float("nan")))
    eta = float(params.get("eta", float("nan")))
    gamma = float(params.get("gamma", float("nan")))
    assert math.isfinite(beta), "β 不是有限值：%r" % beta
    assert math.isfinite(eta), "η 不是有限值：%r" % eta
    assert math.isfinite(gamma), "γ 不是有限值：%r" % gamma
    assert abs(beta - exp["beta"]) <= 0.0025 * abs(exp["beta"]), \
        "β 偏差 %.4f 超容差（基准 %.4f）" % (beta - exp["beta"], exp["beta"])
    assert abs(eta - exp["eta"]) <= 0.0025 * abs(exp["eta"]), \
        "η 偏差 %.4f 超容差（基准 %.4f）" % (eta - exp["eta"], exp["eta"])
    assert abs(gamma - exp["gamma"]) <= max(0.02 * abs(exp["gamma"]), 1.0), \
        "γ 偏差 %.4f 超容差（基准 %.4f）" % (gamma - exp["gamma"], exp["gamma"])

    gamma = float(report["params"]["gamma"])
    eta = float(report["params"]["eta"])
    assert gamma <= 0.15 * eta, "两参数批次却估出大 γ：%.1f（η=%.1f）" % (gamma, eta)


def test_invalid_records_rejected_without_output():
    """题面原文：「stderr 说明原因，不落任何文件」。

    五种坏输入（含边界值 t=0）逐个验证：退出码 2、无报告、无临时/副作用文件、
    已有报告字节不变、stderr 非空。
    """
    good = make_tests(2.0, 800.0, 60.0, n=30, seed=61, censor_time=500.0)
    all_cens = json.loads(json.dumps(good))
    for rec in all_cens["records"]:
        rec["failed"] = False
    dup = json.loads(json.dumps(good))
    dup["records"][1]["id"] = dup["records"][0]["id"]
    zero_time = json.loads(json.dumps(good))
    zero_time["records"][2]["time"] = 0.0
    neg_time = json.loads(json.dumps(good))
    neg_time["records"][2]["time"] = -5.0
    few = json.loads(json.dumps(good))
    few["records"] = few["records"][:3]

    tmp = tempfile.mkdtemp(prefix="wb-neg-")
    keep = os.path.join(tmp, "report.json")
    report, _ = run_case(good)
    assert isinstance(report, dict), "前置的正常记录没跑通"
    with open(keep, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)
    before = open(keep, "rb").read()

    for bad in (all_cens, dup, zero_time, neg_time, few):
        case = tempfile.mkdtemp(prefix="wb-bad-")
        tp = os.path.join(case, "tests.json")
        op = os.path.join(case, "report.json")
        with open(tp, "w", encoding="utf-8") as f:
            json.dump(bad, f, ensure_ascii=False)
        proc = _run(tp, op)
        assert proc.returncode == 2, "坏输入退出码是 %r，应为 2" % proc.returncode
        assert not os.path.exists(op), "坏输入却写出了报告"
        assert (proc.stderr or "").strip(), "坏输入没有给出可读原因"
        leftovers = [n for n in os.listdir(case) if n != "tests.json"]
        assert not leftovers, "坏输入留下了副作用文件：%r" % leftovers

    bad_path = os.path.join(tmp, "bad.json")
    with open(bad_path, "w", encoding="utf-8") as f:
        json.dump(dup, f, ensure_ascii=False)
    proc = _run(bad_path, keep)
    assert proc.returncode == 2
    assert open(keep, "rb").read() == before, "坏输入覆盖了上一次的报告"


def test_search_precision_on_clean_sample():
    """题面原文：「按极大似然拟合三参数 Weibull」——给出的参数必须真的把似然顶到最高。

    干净的大样本（无删失）上，把报告里的参数与独立复算的参数各算一遍对数似然：
    报告值不能比独立复算差（允许 1e-6 的数值噪声），并且对 β、η 各扰动 2% 之后似然必须下降。
    """
    data = make_tests(2.1, 1000.0, 0.0, n=120, seed=71, censor_time=None)
    report, _ = run_case(data)
    assert isinstance(report, dict), "没有产出报告"
    exp = expect(data)
    beta = float((report.get("params") or {}).get("beta", float("nan")))
    eta = float((report.get("params") or {}).get("eta", float("nan")))
    gamma = float((report.get("params") or {}).get("gamma", float("nan")))
    assert math.isfinite(beta) and math.isfinite(eta) and math.isfinite(gamma), "参数不是有限值"
    best = ln_likelihood(beta, eta, gamma, data)
    ref = ln_likelihood(exp["beta"], exp["eta"], exp["gamma"], data)
    assert best >= ref - 1e-6, "报告的似然 %.8f 比独立复算 %.8f 差，说明没搜到极大点" % (best, ref)
    assert best > ln_likelihood(beta * 1.02, eta, gamma, data), "把 β 抬高 2% 似然反而更高"
    assert best > ln_likelihood(beta * 0.98, eta, gamma, data), "把 β 压 2% 似然反而更高"
    assert best > ln_likelihood(beta, eta * 1.02, gamma, data), "把 η 抬高 2% 似然反而更高"
    assert best > ln_likelihood(beta, eta * 0.98, gamma, data), "把 η 压 2% 似然反而更高"


def test_report_schema_and_rounding():
    """题面原文：「三个参数各留三位小数，B 寿命留一位」

    校验顶层字段、计数、位数与有限性，文件里不出现 NaN 或 Infinity。
    """
    data = make_tests(2.2, 850.0, 80.0, n=36, seed=67, censor_time=700.0)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没有产出报告"
    for key in ("n", "failures", "censored", "params", "b_life", "b_life_percent", "loglik", "meta"):
        assert key in report, "报告缺字段 %s" % key
    assert report["n"] == len(data["records"])
    fails = sum(1 for r in data["records"] if r["failed"])
    assert report["failures"] == fails and report["censored"] == len(data["records"]) - fails
    for key in ("beta", "eta", "gamma"):
        v = float(report["params"][key])
        assert math.isfinite(v) and abs(v - round(v, 3)) <= 1e-12, "参数 %s 不是三位小数" % key
    assert abs(float(report["b_life"]) - round(float(report["b_life"]), 1)) <= 1e-9
    assert abs(float(report["loglik"]) - round(float(report["loglik"]), 6)) <= 1e-12
    raw = open(ctx["out"], encoding="utf-8").read()
    assert "NaN" not in raw and "Infinity" not in raw


def test_rerun_produces_identical_bytes():
    """题面原文：「同一份记录重复复核要给出逐字节相同的结果」。

    连跑两次，两个报告文件字节完全相同。
    """
    data = make_tests(2.35, 820.0, 95.0, n=48, seed=11, censor_time=700.0)
    first, ctx1 = run_case(data)
    assert isinstance(first, dict), "第一次没跑通"
    body1 = open(ctx1["out"], "rb").read()
    second, ctx2 = run_case(data)
    assert isinstance(second, dict), "第二次没跑通"
    assert body1 == open(ctx2["out"], "rb").read(), "两次运行的报告字节不一致"


def test_failed_run_keeps_previous_report():
    """题面原文：「失败的运行不得动到已经存在的结论文件」。

    先跑一次合格记录留下报告，再用重复 id 的坏记录跑：退出码 2，旧报告字节不变。
    """
    good = make_tests(2.0, 900.0, 70.0, n=30, seed=71, censor_time=600.0)
    tmp = tempfile.mkdtemp(prefix="wb-keep-")
    tp = os.path.join(tmp, "tests.json")
    op = os.path.join(tmp, "report.json")
    with open(tp, "w", encoding="utf-8") as f:
        json.dump(good, f, ensure_ascii=False)
    first = _run(tp, op)
    assert first.returncode == 0 and os.path.exists(op), "前置的正常记录没跑通"
    before = open(op, "rb").read()
    broken = json.loads(json.dumps(good))
    broken["records"][1]["id"] = broken["records"][0]["id"]
    with open(tp, "w", encoding="utf-8") as f:
        json.dump(broken, f, ensure_ascii=False)
    second = _run(tp, op)
    assert second.returncode == 2
    assert open(op, "rb").read() == before, "失败的那次覆盖了旧报告"


def test_large_sample_recovers_parameters():
    """题面原文：「样本量大了以后，估出来的三个参数要回到真值附近」。

    300 件的批次（约三成删失）：β 与真值相差不超过 10%、η 不超过 6%，γ 落在真值一半到一倍半之间。
    """
    truth = {"beta": 2.0, "eta": 1000.0, "gamma": 150.0}
    data = make_tests(truth["beta"], truth["eta"], truth["gamma"], n=300, seed=83,
                      censor_time=800.0, censor_every=3)
    report, _ = run_case(data)
    assert isinstance(report, dict), "没有产出报告"
    p = report.get("params") or {}
    beta, eta, gamma = float(p["beta"]), float(p["eta"]), float(p["gamma"])
    assert abs(beta - truth["beta"]) <= 0.10 * truth["beta"], "大样本 β 偏差过大：%.3f" % beta
    assert abs(eta - truth["eta"]) <= 0.06 * truth["eta"], "大样本 η 偏差过大：%.1f" % eta
    assert 0.5 * truth["gamma"] <= gamma <= 1.5 * truth["gamma"], "大样本 γ 不合理：%.1f" % gamma


def test_zero_survival_time_rejected():
    """题面原文：「整份记录就是非法输入」。

    把一条记录的时间改成 0（边界值）：必须退出码 2 且不产出结论文件。
    """
    data = make_tests(2.0, 900.0, 50.0, n=40, seed=811, censor_time=None)
    data["records"][0]["time"] = 0.0
    tmp = tempfile.mkdtemp(prefix="wb-bound-")
    tpath = os.path.join(tmp, "tests.json")
    opath = os.path.join(tmp, "report.json")
    with open(tpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    proc = _run(tpath, opath)
    assert proc.returncode == 2, "时间为 0 时退出码应为 2，实际 %r" % proc.returncode
    assert not os.path.exists(opath), "时间为 0 时不该产出结论"

