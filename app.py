"""
app.py — LLM Fine-tuning Lab
Streamlit UI for:
  - Launching fine-tuning jobs
  - Comparing base vs fine-tuned model responses
  - Viewing evaluation metrics and charts
"""

import streamlit as st
import json
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="LLM Fine-tuning Lab", page_icon="🧠", layout="wide")

with st.sidebar:
    st.title("🧠 Fine-tuning Lab")
    hf_token = st.text_input("HuggingFace Token", type="password", value=os.getenv("HF_TOKEN", ""))
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    st.divider()
    base_model = st.selectbox("Base Model", [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/phi-2",
        "mistralai/Mistral-7B-v0.1",
    ])
    os.environ["BASE_MODEL"] = base_model

    st.divider()
    st.markdown("**LoRA Hyperparameters**")
    lora_r = st.slider("LoRA Rank (r)", 4, 64, 16)
    lora_alpha = st.slider("LoRA Alpha", 8, 128, 32)
    max_steps = st.slider("Training Steps", 10, 500, 100)
    lr = st.select_slider("Learning Rate", options=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4], value=2e-4)

    os.environ["LORA_R"] = str(lora_r)
    os.environ["LORA_ALPHA"] = str(lora_alpha)
    os.environ["MAX_STEPS"] = str(max_steps)
    os.environ["LEARNING_RATE"] = str(lr)

tab1, tab2, tab3 = st.tabs(["🚀 Train", "🔬 Compare Models", "📊 Evaluation Metrics"])

# ── Tab 1: Training ─────────────────────────────────────────────────────────
with tab1:
    st.header("Fine-tune a Biomedical LLM with LoRA")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**What is LoRA?**")
        st.info("""LoRA adds small trainable matrices (rank r) to frozen model weights.
Only 0.1–1% of parameters are trained, reducing GPU memory by ~70%
while achieving performance close to full fine-tuning.""")
    with col2:
        params_trained = lora_r * 2 * 768 * 2  # rough estimate
        st.metric("Approx. trainable params", f"{params_trained:,}")
        st.metric("Est. training time (CPU)", f"~{max_steps // 5} min")
        st.metric("Dataset", "PubMedQA (biomedical QA)")

    st.divider()

    prep_col, train_col = st.columns(2)

    with prep_col:
        if st.button("📥 Prepare Dataset", use_container_width=True):
            with st.spinner("Downloading PubMedQA dataset..."):
                try:
                    import sys
                    sys.path.insert(0, ".")
                    from data.prepare_dataset import load_and_prepare
                    ds = load_and_prepare()
                    st.success(f"Dataset ready: {len(ds['train'])} train / {len(ds['validation'])} val / {len(ds['test'])} test")
                    st.json({"sample": ds["train"][0]["text"][:200] + "..."})
                except Exception as e:
                    st.warning("Using synthetic demo dataset (PubMedQA download failed)")
                    st.caption(str(e))

    with train_col:
        if st.button("🚀 Start Fine-tuning", type="primary", use_container_width=True):
            if not hf_token:
                st.error("Add your HuggingFace token in the sidebar.")
            else:
                with st.spinner(f"Fine-tuning {base_model} for {max_steps} steps... (this takes several minutes)"):
                    try:
                        from src.finetune import train, FineTuningConfig
                        cfg = FineTuningConfig(
                            base_model=base_model,
                            lora_r=lora_r,
                            lora_alpha=lora_alpha,
                            max_steps=max_steps,
                            learning_rate=lr,
                        )
                        model_path = train(cfg)
                        st.success(f"Training complete! Model saved to: {model_path}")
                        st.session_state["model_path"] = model_path
                    except Exception as e:
                        st.error(f"Training failed: {e}")

# ── Tab 2: Model Comparison ──────────────────────────────────────────────────
with tab2:
    st.header("Base vs Fine-tuned Model Comparison")

    custom_q = st.text_area("Enter a biomedical question:", height=80,
        value="Does high seizure frequency correlate with cognitive decline in epilepsy patients?")
    custom_ctx = st.text_area("Context (optional):", height=60,
        value="Longitudinal studies of epilepsy patients show varying cognitive outcomes based on seizure control.")

    if st.button("Compare Models", type="primary"):
        ft_path = st.session_state.get("model_path", "outputs/checkpoints/final_model")
        if not Path(ft_path).exists():
            st.warning("Fine-tuned model not found. Showing demo comparison.")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Base Model**")
                st.info("I cannot provide specific medical advice. Please consult a qualified healthcare professional for questions about epilepsy and cognitive effects.")
            with col2:
                st.markdown("**Fine-tuned Model**")
                st.success("Yes, research demonstrates that higher seizure frequency is significantly associated with greater cognitive decline in epilepsy patients, particularly in cases of drug-resistant epilepsy where seizures remain poorly controlled.")
        else:
            with st.spinner("Running both models..."):
                try:
                    from src.inference import load_model_for_inference, generate_answer
                    base_model_obj, base_tok = load_model_for_inference(base_model)
                    ft_model_obj, ft_tok = load_model_for_inference(ft_path)

                    base_ans = generate_answer(base_model_obj, base_tok, custom_q, custom_ctx)
                    ft_ans = generate_answer(ft_model_obj, ft_tok, custom_q, custom_ctx)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Base Model**")
                        st.info(base_ans)
                    with col2:
                        st.markdown("**Fine-tuned Model**")
                        st.success(ft_ans)
                except Exception as e:
                    st.error(str(e))

    st.divider()
    results_path = "outputs/comparison_results.json"
    if Path(results_path).exists():
        with open(results_path) as f:
            saved = json.load(f)
        st.markdown(f"**Saved comparisons: {len(saved)} questions**")
        for i, r in enumerate(saved[:5]):
            with st.expander(f"Q{i+1}: {r['question'][:70]}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Base**"); st.info(r.get("base_model_answer", ""))
                with c2:
                    st.markdown("**Fine-tuned**"); st.success(r.get("finetuned_answer", ""))

# ── Tab 3: Evaluation Metrics ────────────────────────────────────────────────
with tab3:
    st.header("Evaluation Metrics")

    eval_path = "outputs/evaluation_report.json"

    if st.button("Run Evaluation", type="primary"):
        results_path = "outputs/comparison_results.json"
        if Path(results_path).exists():
            with st.spinner("Computing metrics..."):
                from src.evaluate import compare_and_report
                df = compare_and_report(results_path)
                st.session_state["eval_df"] = df
        else:
            st.warning("No comparison results found. Using demo metrics.")
            demo = {
                "metric": ["rouge1","rouge2","rougeL","bleu","exact_match","biomedical_coverage"],
                "base_model": [0.18, 0.05, 0.14, 0.06, 0.33, 0.12],
                "finetuned_model": [0.42, 0.21, 0.38, 0.18, 0.67, 0.31],
            }
            st.session_state["eval_df"] = pd.DataFrame(demo)

    if "eval_df" in st.session_state:
        df = st.session_state["eval_df"]
        st.dataframe(df, use_container_width=True)

        metrics = df["metric"].tolist() if "metric" in df.columns else df.index.tolist()
        base_vals = df["base_model"].tolist()
        ft_vals = df["finetuned_model"].tolist()

        fig = go.Figure(data=[
            go.Bar(name="Base Model", x=metrics, y=base_vals, marker_color="#636EFA"),
            go.Bar(name="Fine-tuned", x=metrics, y=ft_vals, marker_color="#00CC96"),
        ])
        fig.update_layout(
            barmode="group", title="Base vs Fine-tuned Model Metrics",
            yaxis_title="Score", xaxis_title="Metric",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        improvements = [ft - b for ft, b in zip(ft_vals, base_vals)]
        fig2 = px.bar(
            x=metrics, y=improvements,
            color=improvements, color_continuous_scale="RdYlGn",
            title="Improvement: Fine-tuned vs Base",
            labels={"x": "Metric", "y": "Delta"},
        )
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
