"""Figures from processed tables (+ raw timelines).   python -m hddp.figures --stage final  -> results/figures/

Error bars are 95% t-confidence intervals across independent trials (launches). Points with fewer than two
trials have no interval and are drawn without an error bar; the caption then says so.
"""
import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RAW, FIG = ROOT / "results" / "processed", ROOT / "results" / "raw", ROOT / "results" / "figures"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical order (blue, orange, aqua, yellow)
SEQ5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # ordinal blue ramp (light -> dark)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SETTINGS = ["2gpu_nvlink", "2gpu_pcie", "4gpu"]
# non-NVLink paths use NCCL's shared-memory transport through host memory (NCCL_P2P_LEVEL=NVL), not PCIe P2P
SET_LABEL = {"2gpu_nvlink": "2 GPU, NVLink", "2gpu_pcie": "2 GPU, no NVLink (SHM via host)",
             "4gpu": "4 GPU, NVLink + SHM"}
MODEL_LABEL = {"resnet50": "ResNet-50", "vit_b_16": "ViT-B/16"}
ALGOS = ["Auto", "Ring", "Tree"]

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 6, "font.size": 9,
    "axes.titlesize": 10, "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 150,
})


def load(stage, tag=""):
    rows = []
    with open(PROC / f"{stage}_conditions.csv", newline="") as f:
        table = list(csv.DictReader(f))
    for r in table:
        for k, v in r.items():
            if v == "":
                r[k] = None
            else:
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
        if r["instrument"] == 1 and (r["tag"] or "") == tag:
            rows.append(r)
    return rows


def pick(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def series(ax, rows, x, y, color, label, ls="-", marker="o", xkey_mean=True):
    xv = lambda r: r[f"{x}_mean"] if xkey_mean and r.get(f"{x}_mean") is not None else r[x]
    rows = sorted([r for r in rows if r.get(f"{y}_mean") is not None], key=xv)
    if not rows:
        return None
    xs = [xv(r) for r in rows]
    ys = [r[f"{y}_mean"] for r in rows]
    es = [r.get(f"{y}_ci95") or 0 for r in rows]  # no interval (n < 2) -> no error bar
    ax.errorbar(xs, ys, yerr=es, color=color, label=label, ls=ls, marker=marker, capsize=3, elinewidth=1,
                markeredgecolor="#fcfcfb", markeredgewidth=1)
    return min(int(r.get(f"{y}_n") or 0) for r in rows)


def save(fig, name, note=None, legend_ax=None, handles=None):
    if note:  # wrap the caption to the figure width (about 18 characters per inch at 7 pt)
        width = int(fig.get_figwidth() * 18)
        note = "\n".join(textwrap.fill(line, width) for line in note.split("\n"))
    bottom = (0.11 * (note.count("\n") + 1) + 0.02) / fig.get_figheight() if note else 0.0  # room for the note
    if legend_ax is not None or handles is not None:
        hs, ls = (handles, [h.get_label() for h in handles]) if handles else legend_ax.get_legend_handles_labels()
        fig.legend(hs, ls, loc="lower center", ncol=min(len(hs), 5), bbox_to_anchor=(0.5, bottom))
        bottom += 0.09 if fig.get_figheight() < 4.5 else 0.05
    if note:
        fig.text(0.01, 0.005, note, fontsize=7, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, bottom, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def ci_note(ns):
    """Caption fragment stating what the error bars are, from the per-point trial counts actually plotted."""
    ns = [n for n in ns if n is not None]
    if not ns:
        return ""
    lo, hi = min(ns), max(ns)
    n = f"n = {lo}" if lo == hi else f"n = {lo}-{hi}"
    extra = " Points with n = 1 have no error bar." if lo < 2 else ""
    return f"Error bars: 95% t-CI over independent launches ({n}).{extra}"


def join_notes(*parts):
    return "\n".join(p for p in parts if p)


def by_delay(rows, y, ylabel, name, ref=None, note=None, bucket=25.0, control=None):
    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.5), squeeze=False)
    ns = []
    for ax, m in zip(axes[0], models):
        if control:
            ns.append(series(ax, pick(control, model=m, setting="4gpu", algo="Auto", bucket=bucket),
                             "delay_req" if y != "step_ms" else "delay_meas_ms", y, CAT[2],
                             "4 GPU, delay on rank 0", ls="--", marker="s"))
        for i, s in enumerate(SETTINGS):
            rs = pick(rows, model=m, setting=s, algo="Auto", bucket=bucket)
            ns.append(series(ax, rs, "delay_req" if y != "step_ms" else "delay_meas_ms", y, CAT[i],
                             SET_LABEL[s] + (", delay on rank 3" if control and s == "4gpu" else "")))
            if ref == "A1" and y == "step_ms" and rs:
                r0 = min(rs, key=lambda r: r["delay_req"])
                xs = [r["delay_meas_ms_mean"] for r in rs]
                ax.plot([0, max(xs)], [r0["step_ms_mean"], r0["step_ms_mean"] + max(xs)], ls=":", lw=1,
                        color=CAT[i])
        if ref == "one":
            ax.axhline(1.0, color=INK2, lw=1, ls="--")
        ax.set_title(MODEL_LABEL[m])
        ax.set_xlabel("measured injected delay δ (ms)" if y == "step_ms" else "requested delay (ms)")
        ax.set_ylabel(ylabel)
    save(fig, name, join_notes(note, ci_note(ns)), legend_ax=axes[0][0])


def auto_tree_share(setting="4gpu"):
    """{(model, bucket): fraction of logged gradient all-reduces for which NCCL Auto chose Tree} from the
    NCCL verification logs (nccl_verification.csv)."""
    p = PROC / "nccl_verification.csv"
    if not p.exists():
        return {}
    out = {}
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            if r["setting"] == setting and r["algo_requested"] == "Auto":
                sel = json.loads(r["allreduce_selected"])
                tot = sum(sel.values())
                out[(r["model"], float(r["bucket"]))] = sum(v for k, v in sel.items() if k.startswith("Tree")) / tot
    return out


def mark_algorithm_switch(ax, model, setting, algo):
    """Shade bucket sizes at which NCCL Auto chose Tree for most bucket all-reduces: there, a bucket-size effect
    is confounded with a change of the collective algorithm. Returns a caption fragment."""
    if algo != "Auto":
        return ""
    hits = sorted((b, f) for (m, b), f in auto_tree_share(setting).items() if m == model and f > 0.5)
    for b, _ in hits:
        ax.axvspan(b / 1.25, b * 1.25, color=GRID, alpha=0.7, zorder=0, lw=0)
        ax.annotate("Auto → Tree", (b, 1), xycoords=("data", "axes fraction"), ha="center", va="top",
                    fontsize=7, color=INK2)
    others = [f for (m, b), f in auto_tree_share(setting).items() if m == model and f <= 0.5]
    return "".join(f"{MODEL_LABEL[model]}, NCCL Auto: at {b:g} MB NCCL chose Tree for {100 * f:.0f}% of logged "
                   f"gradient all-reduces (≤ {100 * max(others, default=0):.0f}% at the other sizes), so the "
                   f"shaded point mixes a bucket-size and an algorithm effect." for b, f in hits)


def by_bucket(rows, y, ylabel, name, setting="4gpu", algo="Auto"):
    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.3), squeeze=False)
    delays = sorted({r["delay_req"] for r in rows})
    ns, notes = [], []
    for ax, m in zip(axes[0], models):
        for i, d in enumerate(delays):
            ns.append(series(ax, pick(rows, model=m, setting=setting, algo=algo, delay_req=d), "bucket", y,
                             SEQ5[i % len(SEQ5)], f"δ={d:g} ms", xkey_mean=False))
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 25, 64, 128], ["1", "4", "16", "25", "64", "128"])
        ax.set_title(f"{MODEL_LABEL[m]} — {SET_LABEL[setting]}, NCCL {algo}")
        ax.set_xlabel("DDP bucket_cap_mb")
        ax.set_ylabel(ylabel)
        notes.append(mark_algorithm_switch(ax, m, setting, algo))
    save(fig, name, join_notes(*notes, ci_note(ns)), legend_ax=axes[0][0])


def algo_cmp(rows, name, setting="4gpu"):
    models = sorted({r["model"] for r in rows})
    delays = sorted({r["delay_req"] for r in rows})
    dsel = [delays[0], delays[-1]]
    fig, axes = plt.subplots(2, len(models), figsize=(4.2 * len(models), 6), squeeze=False)
    ns = []
    for j, m in enumerate(models):
        for i, d in enumerate(dsel):
            ax = axes[i][j]
            for k, a in enumerate(ALGOS):
                ns.append(series(ax, pick(rows, model=m, setting=setting, algo=a, delay_req=d), "bucket",
                                 "throughput", CAT[k], a, xkey_mean=False))
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 4, 16, 25, 64, 128], ["1", "4", "16", "25", "64", "128"])
            ax.set_title(f"{MODEL_LABEL[m]} — δ={d:g} ms ({SET_LABEL[setting]})")
            ax.set_xlabel("DDP bucket_cap_mb")
            ax.set_ylabel("throughput (img/s)")
    save(fig, name, join_notes("Auto = NCCL's own choice (Ring for most bucket all-reduces here; see "
                               "nccl_verification.csv). Ring/Tree = forced with NCCL_ALGO; protocol not forced.",
                               ci_note(ns)), legend_ax=axes[0][0])


def homo_vs_het(stage, name):
    summ = json.loads((PROC / f"{stage}_summary.json").read_text())["H3_throughput"]
    rows = load(stage)
    groups = sorted({(h["setting"], h["model"]) for h in summ})
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(4.4 * len(groups), 3.6), squeeze=False)
    get = lambda s, m, cfg, d: next(iter(pick(rows, setting=s, model=m, algo=cfg.split("/")[0],
                                              bucket=float(cfg.split("/")[1][:-2]), delay_req=d)), None)
    for ax, (s, m) in zip(axes[0], groups):
        hs = sorted([h for h in summ if h["setting"] == s and h["model"] == m], key=lambda h: h["delay_req"])
        homo = hs[0]["homo_opt"]
        for k, (lab, cfgs, col, mk) in enumerate((
                ("homogeneous-optimal config (fixed; named in title)", [homo] * len(hs), CAT[0], "o"),
                ("best config at that δ (label)", [h["best_at_delay"] for h in hs], CAT[1], "D"))):
            pts = [(h["delay_req"], get(s, m, c, h["delay_req"]), c) for h, c in zip(hs, cfgs)]
            pts = [(d, r, c) for d, r, c in pts if r]
            ax.errorbar([d for d, _, _ in pts], [r["throughput_mean"] for _, r, _ in pts],
                        yerr=[r.get("throughput_ci95") or 0 for _, r, _ in pts], color=col, marker=mk, label=lab,
                        capsize=3, elinewidth=1, markeredgecolor="#fcfcfb", ls="-" if k == 0 else ":")
            if k == 1:
                for d, r, c in pts:
                    ax.annotate(c + ("" if c != homo else " (same)"), (d, r["throughput_mean"]),
                                textcoords="offset points", xytext=(4, 6), fontsize=7, color=INK2)
        ax.set_xlabel("injected delay (ms)")
        ax.set_ylabel("throughput (img/s)")
        ax.set_title(f"{MODEL_LABEL[m]} — {SET_LABEL[s]}\nhomogeneous-optimal: {homo}", fontsize=9)
    save(fig, name, "Configs = (NCCL algorithm / bucket_cap_mb). Error bars: 95% CI over launches (n = 4-5).\n"
         "The best configuration at each δ is selected from the same data (winner's curse); it is not "
         "independently validated.", legend_ax=axes[0][0])


def timeline(stage, name, setting="4gpu", bucket=25):
    from hddp.analyze import attempt_flags
    with open(RAW / f"{stage}.jsonl") as f:
        rows = [json.loads(line) for line in f]
    good = attempt_flags(rows)["good"]  # never draw a timeline from an excluded attempt
    recs = [r for r in rows if r["type"] == "cond" and r.get("setting") == setting and r["bucket"] == bucket
            and (r["launch_id"], r.get("attempt")) in good and any("timeline" in k for k in r["ranks"])]
    if not recs:
        print("no timeline data")
        return
    models = sorted({r["model"] for r in recs})
    delays = sorted({r["delay_req"] for r in recs})
    fig, axes = plt.subplots(len(delays), len(models), figsize=(6 * len(models), 1.6 + 1.1 * len(delays) * 2),
                             squeeze=False)
    colors = {"c": CAT[0], "n": CAT[1], "d": CAT[3]}
    for j, m in enumerate(models):
        for i, d in enumerate(delays):
            ax = axes[i][j]
            r = next((x for x in recs if x["model"] == m and x["delay_req"] == d), None)
            if r is None:
                continue
            t0 = min(k["timeline_t0_us"] for k in r["ranks"])
            step = sum(k["steps"]["period"][0] for k in r["ranks"]) / len(r["ranks"]) * 1000  # us
            for k in r["ranks"]:
                off = k["timeline_t0_us"] - t0
                for lane, cls in enumerate(("c", "n", "d")):
                    iv = [((s + off) / 1000, (e - s) / 1000) for c, s, e, _ in k["timeline"]
                          if c == cls and s + off < 2.2 * step]
                    y = k["rank"] * 4 + lane
                    ax.broken_barh(iv, (y, 0.85), color=colors[cls])
            role = (f" (delayed {d:g} ms)" if d else " (delay rank, δ = 0)")
            ax.set_yticks([k * 4 + 1.4 for k in range(r["world"])],
                          [f"rank {k}" + (role if k == r["straggler_rank"] else "") for k in range(r["world"])])
            ax.set_xlim(0, 2.2 * step / 1000)
            ax.invert_yaxis()
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{MODEL_LABEL[m]} — {SET_LABEL[setting]}, {bucket} MB, Auto, δ={d:g} ms")
            ax.set_xlabel("time (ms, profiler window)")
    from matplotlib.patches import Patch
    save(fig, name, "Lanes per rank: compute / NCCL / injected delay. torch.profiler (CUPTI) kernel intervals, host "
         "clock.\nOne profiled window of one launch (trial 0) per panel: an illustration, not an average. "
         "NCCL lanes include time NCCL kernels spend waiting for peers.",
         handles=[Patch(color=colors["c"], label="compute kernels"), Patch(color=colors["n"], label="NCCL kernels"),
                  Patch(color=colors["d"], label="injected delay (spin)")])


def slack_scatter(stage, name):
    with open(PROC / f"{stage}_trials.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("A") and r.get("A_slack_model") and r["instrument"] == "1"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    for i, m in enumerate(sorted({r["model"] for r in rows})):
        rs = [r for r in rows if r["model"] == m]
        ax.scatter([float(r["A_slack_model"]) for r in rs], [float(r["A"]) for r in rs], s=22, color=CAT[i],
                   edgecolor="#fcfcfb", linewidth=0.8, label=MODEL_LABEL[m], alpha=0.85)
    lim = [min(-0.1, ax.get_xlim()[0]), max(1.15, ax.get_ylim()[1])]
    ax.plot(lim, lim, color=INK2, lw=1, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("slack-model prediction  max(0, δ − slack) / δ")
    ax.set_ylabel("measured A(δ)")
    ax.set_title("Exploratory post-hoc slack model vs. measured A")
    save(fig, name, "slack = delayed rank's δ=0 wait − least-waiting rank's wait. Dashed: y = x.\n"
         "Formulated after the pilot; descriptive, not a calibrated or causal model. One point per trial.",
         legend_ax=ax)


def main():
    global FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="final")
    ap.add_argument("--outdir", default=str(FIG))
    args = ap.parse_args()
    st = args.stage
    FIG = Path(args.outdir)
    rows = load(st)
    delay = "Delay = calibrated GPU busy-wait (torch.cuda._sleep) before each forward pass on one rank."
    energy = "GPU-board energy from NVML, summed over the GPUs of the job; excludes CPU, DRAM, network and storage."
    by_delay(rows, "step_ms", "step time (ms)", "fig01_step_time_vs_delay", ref="A1",
             note=f"Dotted: step(0)+δ, i.e. A(δ)=1. NCCL Auto, 25 MB buckets. {delay}")
    by_delay(rows, "throughput", "throughput (img/s)", "fig02_throughput_vs_delay",
             note="NCCL Auto, 25 MB buckets.")
    ctrl = load(st, tag="strag0")
    by_delay(rows, "A", "amplification A(δ)", "fig03_amplification_vs_delay", ref="one", control=ctrl,
             note="A(δ) = (step(δ) − step(0)) / measured δ, paired within each launch. Dashed: A = 1 "
                  "(delay passes through one-for-one).")
    by_delay(rows, "energy_J_step", "GPU-board energy (J/step, sum over GPUs)", "fig04a_energy_vs_delay",
             note=energy)
    by_delay(rows, "energy_idle_sub_J_step", "GPU-board energy above idle (J/step)",
             "fig04b_energy_idle_sub_vs_delay",
             note="Measured GPU-board energy minus each GPU's idle power × window time. Includes compute, "
                  "communication and waiting above idle; it is not a measure of communication energy.")
    by_delay(rows, "overlap_ratio_fast", "fraction of NCCL-kernel time\nconcurrent with compute",
             "fig05_overlap_vs_delay",
             note="Fraction of NCCL-kernel residency (including waiting for peers) during which a non-NCCL, non-delay "
                  "kernel runs on the same GPU; mean over non-delayed ranks, from torch.profiler. Not a measure of "
                  "hidden data transfer.")
    by_bucket(rows, "throughput", "throughput (img/s)", "fig06_bucket_vs_throughput")
    by_bucket(rows, "energy_J_step", "GPU-board energy (J/step, sum over GPUs)", "fig07_bucket_vs_energy")
    by_bucket(rows, "A", "amplification A(δ)", "fig07b_bucket_vs_amplification")
    algo_cmp(rows, "fig08_algo_comparison")
    homo_vs_het(st, "fig09_homo_vs_het_optimal")
    timeline(st, "fig10_timeline")
    slack_scatter(st, "fig11_slack_model")


if __name__ == "__main__":
    main()
