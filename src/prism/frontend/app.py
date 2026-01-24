from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st


REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def run(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout, p.stderr


@st.cache_data
def registry_keys() -> Tuple[List[str], List[str]]:
    from prism.experiments.registry import PROCESS_REGISTRY, RECONSTRUCTOR_REGISTRY
    return sorted(PROCESS_REGISTRY.keys()), sorted(RECONSTRUCTOR_REGISTRY.keys())


def find_runs(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []
    runs = [p for p in results_dir.iterdir() if p.is_dir() and (p / "runs.csv").exists()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def rpath(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def main() -> None:
    st.set_page_config(page_title="PRISM", layout="wide")
    st.title("PRISM")
    st.caption(f"repo: `{REPO}`  ·  results: `{REPO/'results'}`")

    results_root = REPO / "results"

    st.sidebar.header("Run")

    try:
        procs, recons = registry_keys()
        process = st.sidebar.selectbox("process", procs, index=procs.index("even_process") if "even_process" in procs else 0)
        reconstructor = st.sidebar.selectbox("reconstructor", recons, index=recons.index("one_step") if "one_step" in recons else 0)
    except Exception:
        # if imports fail for some reason, don't die, just let the user type
        process = st.sidebar.text_input("process", value="even_process")
        reconstructor = st.sidebar.text_input("reconstructor", value="one_step")

    ks = st.sidebar.multiselect("ks", options=list(range(1, 11)), default=[2, 3, 4, 5])
    ks_extra = st.sidebar.text_input("extra ks", value="")
    if ks_extra.strip():
        ks += [int(x) for x in ks_extra.split()]
    ks = sorted(set(ks))

    seeds = st.sidebar.multiselect("seeds", options=list(range(0, 10)), default=[0, 1, 2, 3, 4])
    seeds_extra = st.sidebar.text_input("extra seeds", value="")
    if seeds_extra.strip():
        seeds += [int(x) for x in seeds_extra.split()]
    seeds = sorted(set(seeds))

    length = st.sidebar.number_input("length", min_value=1_000, max_value=50_000_000, value=200_000, step=10_000)
    train_frac = st.sidebar.slider("train_frac", 0.1, 0.95, 0.8, 0.05)

    st.sidebar.subheader("Perturbations (single or sweep)")

    flip_ps_str = st.sidebar.text_input("flip_ps (optional)", value="")
    sub_steps_str = st.sidebar.text_input("subsample_steps (optional)", value="")

    flip_p = st.sidebar.number_input("flip_p (single)", 0.0, 1.0, 0.0, 0.01)
    sub_step = st.sidebar.number_input("subsample_step (single)", 1, 1024, 1, 1)

    flip_ps = None
    sub_steps = None
    if flip_ps_str.strip():
        flip_ps = [float(x) for x in flip_ps_str.split()]
    if sub_steps_str.strip():
        sub_steps = [int(x) for x in sub_steps_str.split()]

    st.sidebar.subheader("Representation")
    noisy_rep = st.sidebar.checkbox("noisy representation", value=False)
    noise_seed = st.sidebar.number_input("noise_seed", min_value=0, max_value=10_000_000, value=123, step=1)

    st.sidebar.subheader("Transitions")
    save_transitions = st.sidebar.checkbox("save transitions", value=False)
    rep_names = [f"last_{k}" for k in ks] if ks else ["last_2"]
    show_rep = st.sidebar.selectbox("show_transitions_for", options=rep_names, index=0)
    show_seed = st.sidebar.number_input("show_seed", min_value=0, max_value=10_000_000, value=(seeds[0] if seeds else 0), step=1)

    st.sidebar.subheader("Output")
    outdir = st.sidebar.text_input("outdir", value="results/run_ui")
    force = st.sidebar.checkbox("force overwrite runs.csv", value=True)

    st.sidebar.subheader("Figures")
    make_phase = st.sidebar.checkbox("phase diagrams", value=True)
    plot_sub_step = st.sidebar.number_input("subsample_step for k-plots", 1, 1024, 1, 1)
    plot_metrics = st.sidebar.multiselect(
        "metrics vs k",
        options=["branch_entropy", "unifilarity_score", "logloss", "n_states", "C_mu_empirical"],
        default=["branch_entropy", "unifilarity_score", "logloss"],
    )

    out = REPO / outdir
    out.parent.mkdir(parents=True, exist_ok=True)

    # main buttons
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Run (prism.cli)", type="primary", use_container_width=True):
            if not ks or not seeds:
                st.error("Need ks and seeds.")
            else:
                cmd = [
                    sys.executable, "-m", "prism.cli",
                    "--process", process,
                    "--reconstructor", reconstructor,
                    "--ks", *map(str, ks),
                    "--seeds", *map(str, seeds),
                    "--length", str(int(length)),
                    "--train-frac", str(float(train_frac)),
                    "--outdir", str(out),
                ]
                if force:
                    cmd.append("--force")

                # sweep overrides single
                if flip_ps is not None:
                    cmd += ["--flip-ps", *map(str, flip_ps)]
                else:
                    cmd += ["--flip-p", str(float(flip_p))]

                if sub_steps is not None:
                    cmd += ["--subsample-steps", *map(str, sub_steps)]
                else:
                    cmd += ["--subsample-step", str(int(sub_step))]

                if noisy_rep:
                    cmd += ["--noisy", "--noise-seed", str(int(noise_seed))]

                if save_transitions:
                    cmd += ["--save-transitions", "--show-transitions-for", show_rep, "--show-seed", str(int(show_seed))]

                st.code(" ".join(shlex.quote(x) for x in cmd), language="bash")
                code, stdout, stderr = run(cmd, cwd=REPO)

                if code == 0:
                    st.success("done")
                else:
                    st.error(f"failed (exit {code})")

                with st.expander("stdout"):
                    st.text(stdout)
                with st.expander("stderr"):
                    st.text(stderr)

    with c2:
        if st.button("Summarise", use_container_width=True):
            cmd = [sys.executable, "-m", "prism.analysis.summarise", "--root", str(out)]
            st.code(" ".join(shlex.quote(x) for x in cmd), language="bash")
            code, stdout, stderr = run(cmd, cwd=REPO)
            st.success("done") if code == 0 else st.error(f"failed (exit {code})")
            with st.expander("stdout"):
                st.text(stdout)
            with st.expander("stderr"):
                st.text(stderr)

    with c3:
        if st.button("Make figures", use_container_width=True):
            cmd = [
                sys.executable, "-m", "prism.analysis.make_figures",
                "--root", str(out),
                "--subsample-step", str(int(plot_sub_step)),
                "--metrics", *plot_metrics,
            ]
            if make_phase:
                cmd.append("--phase")

            st.code(" ".join(shlex.quote(x) for x in cmd), language="bash")
            code, stdout, stderr = run(cmd, cwd=REPO)
            st.success("done") if code == 0 else st.error(f"failed (exit {code})")
            with st.expander("stdout"):
                st.text(stdout)
            with st.expander("stderr"):
                st.text(stderr)

    st.divider()

    # browse existing runs
    st.header("Browse")

    run_dirs = find_runs(results_root)
    options = ["(current outdir)"] + [rpath(p, results_root) for p in run_dirs]
    choice = st.selectbox("run dir", options=options)

    run_dir = out if choice == "(current outdir)" else (results_root / choice)
    st.write(f"`{run_dir}`")

    # quick status row
    for name in ["config.json", "runs.csv", "summary_by_condition.csv", "summary_simple.csv"]:
        st.write(f"- {name}: {'✅' if (run_dir/name).exists() else '—'}")

    # preview summary_simple
    ss = run_dir / "summary_simple.csv"
    if ss.exists():
        st.subheader("summary_simple.csv (top 50 lines)")
        st.text("\n".join(ss.read_text(encoding="utf-8").splitlines()[:50]))

    # figures
    figs = run_dir / "figures"
    st.subheader("figures")
    if figs.exists():
        pdfs = sorted(figs.rglob("*.pdf"))
        pngs = sorted(figs.rglob("*.png"))
        if not pdfs and not pngs:
            st.info("no figures yet")
        for p in pdfs:
            st.download_button(f"download {p.name}", data=p.read_bytes(), file_name=p.name, mime="application/pdf")
        for p in pngs[:30]:
            st.image(str(p), caption=rpath(p, run_dir), use_container_width=True)
    else:
        st.info("no figures dir")

    # transitions
    tdir = run_dir / "transitions"
    st.subheader("transitions")
    if tdir.exists():
        t_pngs = sorted(tdir.rglob("*.png"))
        if not t_pngs:
            st.info("no transitions yet (run with save_transitions)")
        for p in t_pngs[:40]:
            st.image(str(p), caption=rpath(p, run_dir), use_container_width=True)
    else:
        st.info("no transitions dir")


if __name__ == "__main__":
    main()