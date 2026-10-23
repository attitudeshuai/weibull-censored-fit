# -*- coding: utf-8 -*-
"""pytest 汇总钩子：按权重把测试结果写成 reward。

每条测试占一个权重，权重和 = 1（见 WEIGHTS）；只有 PASS 才拿到该条权重；
不在 WEIGHTS 里的测试名一律不给分。reward 默认写 /logs/verifier/reward.txt，
本地调试可用 CT_REWARD_PATH 覆盖。
"""
import os

WEIGHTS = {
    "test_params_match_independent_mle_on_base_dataset": 0.127912088,
    "test_params_match_second_dataset_with_different_parameters": 0.117252747,
    "test_censoring_handled_under_heavy_censoring": 0.117252747,
    "test_b_life_matches_independent_computation": 0.095934066,
    "test_two_parameter_limit_when_gamma_near_zero": 0.095934066,
    "test_search_precision_on_clean_sample": 0.106593407,
    "test_report_schema_and_rounding": 0.085274725,
    "test_rerun_produces_identical_bytes": 0.074615385,
    "test_failed_run_keeps_previous_report": 0.074615385,
    "test_large_sample_recovers_parameters": 0.074615385,
    "test_zero_survival_time_rejected": 0.030000000,
}

_RESULTS = {}


def pytest_runtest_logreport(report):
    name = report.nodeid.split("::")[-1]
    if report.when in ("setup", "call") and report.failed:
        _RESULTS[name] = False
    elif report.when == "call" and report.passed:
        _RESULTS.setdefault(name, True)


def pytest_sessionfinish(session, exitstatus):
    # 完成报告校验：必须有每一条测试的结果，缺任何一条都按 INCOMPLETE 记 0
    missing = [n for n in WEIGHTS if n not in _RESULTS]
    total = 0.0
    if missing:
        print("[INCOMPLETE] 这些测试没有跑完：%s；reward 记 0" % ", ".join(missing))
    else:
        for name, weight in WEIGHTS.items():
            if _RESULTS.get(name) is True:
                total += weight
    path = os.environ.get("CT_REWARD_PATH", "/logs/verifier/reward.txt")
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("%.6f\n" % total)
