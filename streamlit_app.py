"""Streamlit version of the GPT-2 Fine-tuning Lab (same backend as the FastAPI site).
Run locally:  streamlit run streamlit_app.py
"""
import re
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from backend import core

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="GPT-2 Fine-tuning Lab", layout="wide")
st.title("GPT-2 Fine-tuning Lab")
st.caption("Upload a small custom dataset, fine-tune GPT-2, then compare base vs fine-tuned generations.")

ss = st.session_state
ss.setdefault("texts", [])
ss.setdefault("last_upload", None)


def set_texts(raw: str):
    ss.texts = core.split_examples(core.clean_text(raw))


# ------------------------------------------------------------------ 1. dataset
st.header("1. Dataset")
c1, c2 = st.columns([3, 1])
up = c1.file_uploader("Upload a .txt file", type=["txt"])
if up is not None and (up.name, up.size) != ss.last_upload:
    ss.last_upload = (up.name, up.size)
    set_texts(up.getvalue().decode("utf-8", errors="ignore"))
if c2.button("Load sample (campus announcements)"):
    set_texts((ROOT / "data" / "campus_announcements.txt").read_text())
pasted = st.text_area("...or paste text (blank line between examples, or one example per line)", height=120)
if st.button("Use pasted text"):
    set_texts(pasted)

if ss.texts:
    s = core.dataset_stats(ss.texts)
    st.success(f"{s['examples']} examples, {s['words']} words")
    st.write("Top keywords: " + ", ".join(s["keywords"]))
    with st.expander("Preview"):
        for t in s["preview"]:
            st.write("- " + t)
else:
    st.info("No dataset loaded yet.")

# ------------------------------------------------------------------ 2. training
st.header("2. Training settings")
a, b, c, d, e, f = st.columns(6)
base = a.selectbox("Base model", ["distilgpt2", "gpt2", core.TINY])
run = b.text_input("Run name", "my_run")
epochs = c.number_input("Epochs", 1, 100, 3)
bs = d.number_input("Batch size", 1, 64, 4)
lr = e.number_input("Learning rate", 1e-6, 1e-2, 5e-5, format="%.6f")
block = f.number_input("Block size", 8, 512, 64)

if st.button("Start training", type="primary", disabled=not ss.texts):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", run):
        st.error("Run name: letters, digits, _ and - only (max 40).")
    else:
        prog = core.Progress()
        cfg = core.TrainConfig(base, run, int(epochs), int(bs), float(lr), int(block))
        texts = list(ss.texts)

        def work():
            try:
                core.train(texts, cfg, prog)
            except Exception as ex:
                prog.status, prog.message = "error", f"{type(ex).__name__}: {ex}"

        th = threading.Thread(target=work, daemon=True)
        th.start()
        bar, info, chart = st.progress(0.0), st.empty(), st.empty()
        while True:
            alive = th.is_alive()
            bar.progress(min(1.0, prog.step / prog.total_steps) if prog.total_steps else 0.0)
            last = f"{prog.train_log[-1]['loss']:.4f}" if prog.train_log else "-"
            info.write(f"**{prog.status}**: {prog.message} | step {prog.step}/{prog.total_steps} | epoch {prog.epoch} | last loss {last}")
            if prog.train_log:
                chart.line_chart(pd.DataFrame(prog.train_log).set_index("step")[["loss"]])
            if not alive:
                break
            time.sleep(0.7)
        if prog.status == "error":
            st.error(prog.message)
        else:
            v0, v1 = prog.val_log[0]["val_loss"], prog.val_log[-1]["val_loss"]
            st.success(f"Done. Validation loss: {v0:.3f} (before) -> {v1:.3f} (after)")
            st.dataframe(pd.DataFrame(prog.val_log), hide_index=True)

# ------------------------------------------------------------------ 3. generation
st.header("3. Generate: base vs fine-tuned")
runs = core.list_runs()
if not runs:
    st.info("Train a model first.")
else:
    g1, g2, g3, g4, g5, g6 = st.columns(6)
    sel = g1.selectbox("Checkpoint", [r["run_name"] for r in runs])
    mnt = g2.number_input("Max new tokens", 1, 200, 50)
    temp = g3.slider("Temperature", 0.1, 2.0, 0.8)
    topk = g4.number_input("Top-k", 0, 500, 50)
    topp = g5.slider("Top-p", 0.05, 1.0, 0.95)
    nsamp = g6.number_input("Samples", 1, 5, 2)
    prompt = st.text_input("Prompt", "Attention students:")
    if st.button("Generate"):
        with st.spinner("Generating..."):
            r = core.compare(sel, prompt, n=int(nsamp), max_new_tokens=int(mnt), temperature=float(temp),
                             top_k=int(topk), top_p=float(topp), seed=0)
        st.write(f"Validation loss on your data: {r['val_loss']['before']:.3f} (before) -> {r['val_loss']['after']:.3f} (after)")
        L, R = st.columns(2)
        for col, key, title in ((L, "base", "Before: base model"), (R, "finetuned", "After: fine-tuned model")):
            m = r[key + "_metrics"]
            col.subheader(title)
            col.caption(f"custom-data word coverage {m['coverage']*100:.0f}% | dataset keyword hits {m['keyword_hits']}")
            for i, t in enumerate(r[key], 1):
                col.markdown(f"**#{i}** {t}")
