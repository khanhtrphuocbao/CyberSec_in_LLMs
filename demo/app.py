"""Local Streamlit evidence board for MedQA V0–V4 benchmark results."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.data import (
    VARIANTS,
    answer_comparison_rows,
    load_variant_results,
    parse_answer_options,
    select_question_result,
    variant_summary,
)
from demo.runner import run_custom_variant, run_variant
from rag.data_loader import MedQALoader


RESULTS_ROOT = ROOT / "results"
SINGLE_ROOT = RESULTS_ROOT / "single_question"
LOG_ROOT = ROOT / "logs" / "single_question"


def _summary_file(variant: str) -> Dict[str, Any]:
    path = RESULTS_ROOT / variant / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _load_results() -> Dict[str, list[Dict[str, Any]]]:
    return load_variant_results(RESULTS_ROOT)


@st.cache_data(show_spinner=False)
def _load_questions(data_path: str) -> list[Dict[str, Any]]:
    return [question.to_dict() for question in MedQALoader().load_json(data_path)]


def _metric_frame(results: Dict[str, list[Dict[str, Any]]]) -> pd.DataFrame:
    metrics = []
    for variant in VARIANTS:
        rows = results.get(variant, [])
        if not rows:
            continue
        summary = variant_summary(rows)
        persisted = _summary_file(variant)
        token_note = persisted.get("token_note")
        average_tokens = None if token_note else sum(
            float(row.get("total_tokens", 0) or 0) for row in rows
        ) / len(rows)
        metrics.append({
            "Variant": variant,
            "Accuracy": summary["accuracy"],
            "Valid rate": summary["valid_rate"],
            "Avg confidence": summary["average_confidence"],
            "Avg latency (s)": summary["average_latency_seconds"],
            "Avg tokens": average_tokens,
            "Token note": token_note or "Per-row telemetry available",
            "Questions": summary["total"],
        })
    return pd.DataFrame(metrics)


def _render_evidence_ribbon(result: Dict[str, Any]) -> None:
    metadata = result.get("metadata") or {}
    stages = [
        ("RAG", bool(metadata.get("rag_used") or metadata.get("rag_trace")), "retrieval"),
        ("Planner", bool(metadata.get("planner_trace")), "planner"),
        ("Examiner", bool(metadata.get("examiner_trace") or result.get("reasoning")), "examiner"),
        ("Evaluator", bool(metadata.get("evaluator_trace")), "evaluator"),
    ]
    cards = "".join(
        f'<span class="ribbon-stage {style} {"active" if enabled else "muted"}">{name}</span>'
        for name, enabled, style in stages
    )
    st.markdown(f'<div class="evidence-ribbon">{cards}</div>', unsafe_allow_html=True)


def _render_dashboard(results: Dict[str, list[Dict[str, Any]]]) -> None:
    st.subheader("Benchmark dashboard")
    frame = _metric_frame(results)
    if frame.empty:
        st.info("Chưa tìm thấy `results/V*/results_V*.json`. Hãy copy artifact benchmark vào thư mục `results/`.")
        return

    cards = st.columns(len(frame))
    for card, row in zip(cards, frame.to_dict("records")):
        card.metric(row["Variant"], f"{row['Accuracy']:.2%}", f"valid {row['Valid rate']:.2%}")
        card.caption(f"{row['Questions']:,} câu · {row['Avg latency (s)']:.1f}s/câu")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        figure = px.bar(
            frame, x="Variant", y="Accuracy", color="Variant", text_auto=".2%",
            color_discrete_sequence=["#0077B6", "#0F766E", "#B45309", "#5B5BD6", "#B42318"],
            title="Accuracy theo variant",
        )
        figure.update_layout(showlegend=False, yaxis_tickformat=".0%", margin=dict(l=12, r=12, t=48, b=12))
        st.plotly_chart(figure, width="stretch")
    with chart_right:
        figure = px.bar(
            frame, x="Variant", y="Avg latency (s)", color="Variant",
            color_discrete_sequence=["#0077B6", "#0F766E", "#B45309", "#5B5BD6", "#B42318"],
            title="Latency trung bình ghi nhận",
        )
        figure.update_layout(showlegend=False, margin=dict(l=12, r=12, t=48, b=12))
        st.plotly_chart(figure, width="stretch")

    st.caption("Token của benchmark lịch sử V2 được đánh dấu non-comparable khi summary báo cumulative telemetry.")
    st.dataframe(
        frame.style.format({
            "Accuracy": "{:.2%}", "Valid rate": "{:.2%}", "Avg confidence": "{:.3f}",
            "Avg latency (s)": "{:.2f}", "Avg tokens": "{:,.0f}",
        }, na_rep="—"),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Drill-down theo kết quả")
    variant = st.selectbox("Variant để lọc", list(results), key="dashboard_variant")
    outcome = st.radio("Kết quả", ["Tất cả", "Đúng", "Sai", "Invalid"], horizontal=True)
    rows = results[variant]
    if outcome == "Đúng":
        rows = [row for row in rows if row.get("is_correct")]
    elif outcome == "Sai":
        rows = [row for row in rows if not row.get("is_correct")]
    elif outcome == "Invalid":
        rows = [row for row in rows if row.get("is_valid") is False]
    table = [{
        "Question": row.get("question_id"), "Predicted": row.get("predicted_answer"),
        "Correct": row.get("correct_answer"), "Correct?": row.get("is_correct"),
        "Confidence": row.get("confidence"), "Latency (s)": row.get("latency_seconds"),
    } for row in rows]
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True, height=320)


def _render_test_answer_comparison(question: Dict[str, Any], variants: list[str]) -> None:
    """Display the persisted predictions and the authoritative test-set answer."""
    rows = answer_comparison_rows(
        question_id=question["question_id"],
        correct_answer=question["answer"],
        variants=variants,
        results_root=RESULTS_ROOT,
        single_root=SINGLE_ROOT,
    )
    st.markdown("#### Kết quả so sánh")
    st.info(f"Đáp án chuẩn của test set: **{question['answer']}**")
    if not rows:
        st.info("Chưa có kết quả mới cho các variant đã chạy.")
        return
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_test_set_runner(question: Dict[str, Any], question_index: int, selected_variants: list[str], top_k: int, two_step: bool) -> None:
    st.subheader(f"Question runner · index {question_index} · {question['question_id']}")
    st.write(question["question"])
    for key, value in question["options"].items():
        st.markdown(f"**{key}.** {value}")

    if not selected_variants:
        st.warning("Chọn ít nhất một variant ở sidebar.")
        return
    st.caption("Các variant chạy tuần tự qua CLI hiện có; API key chỉ được đọc từ `.env`.")
    if st.button("Chạy các variant đã chọn", type="primary"):
        for variant in selected_variants:
            with st.spinner(f"Đang chạy {variant} cho {question['question_id']}…"):
                completed = run_variant(
                    sys.executable, ROOT, variant,
                    question_index=question_index, top_k=top_k, two_step_retrieval=two_step,
                )
            with st.expander(f"{variant} · exit code {completed.returncode}", expanded=completed.returncode != 0):
                st.code((completed.stdout or "") + (completed.stderr or ""), language="text")
            if completed.returncode == 0:
                st.success(f"{variant} đã ghi result/log cho {question['question_id']}.")
            else:
                st.error(f"{variant} không hoàn tất. Xem output ở trên hoặc log trong `{LOG_ROOT}`.")
        _load_results.clear()
        st.session_state["last_test_run"] = {
            "question_id": question["question_id"],
            "variants": list(selected_variants),
        }

    last_run = st.session_state.get("last_test_run")
    if last_run and last_run.get("question_id") == question["question_id"]:
        _render_test_answer_comparison(question, last_run["variants"])


def _render_custom_result_table(results: Dict[str, Dict[str, Any]]) -> None:
    if not results:
        return
    st.markdown("#### Kết quả câu hỏi tuỳ ý")
    st.caption("Câu hỏi tự nhập không có đáp án chuẩn từ test set, nên chỉ hiển thị dự đoán của từng variant.")
    rows = [{
        "Variant": variant,
        "Predicted answer": result.get("predicted_answer"),
        "Valid?": result.get("is_valid"),
        "Confidence": result.get("confidence"),
        "Latency (s)": result.get("latency_seconds"),
        "Total tokens": result.get("total_tokens"),
    } for variant, result in results.items()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_custom_question_runner(selected_variants: list[str], top_k: int, two_step: bool) -> None:
    st.subheader("Question runner · câu hỏi tuỳ ý")
    st.caption("Dán câu hỏi và mỗi lựa chọn trên một dòng; có thể viết `A. ...`, `B) ...` hoặc chỉ nội dung lựa chọn.")
    with st.form("custom_question_form"):
        question_text = st.text_area("Câu hỏi", placeholder="Nhập hoặc dán câu hỏi y khoa…", height=140)
        raw_options = st.text_area(
            "Các đáp án",
            placeholder="A. Lựa chọn thứ nhất\nB. Lựa chọn thứ hai\nC. Lựa chọn thứ ba\nD. Lựa chọn thứ tư",
            height=170,
        )
        submitted = st.form_submit_button(
            "Chạy các variant đã chọn",
            type="primary",
            icon=":material/play_arrow:",
        )

    if submitted:
        if not selected_variants:
            st.warning("Chọn ít nhất một variant ở sidebar.")
            return
        options = parse_answer_options(raw_options)
        if len(options) < 2:
            st.warning("Cần ít nhất hai đáp án không rỗng, mỗi đáp án một dòng.")
            return
        answers: Dict[str, Dict[str, Any]] = {}
        for variant in selected_variants:
            try:
                with st.spinner(f"Đang chạy {variant} cho câu hỏi tuỳ ý…"):
                    answers[variant] = run_custom_variant(
                        variant,
                        question=question_text,
                        options=options,
                        top_k=top_k,
                        two_step_retrieval=two_step,
                    )
                st.success(f"{variant} hoàn tất.")
            except (RuntimeError, ValueError) as error:
                st.error(f"{variant} không hoàn tất: {error}")
        st.session_state["custom_run_results"] = answers

    _render_custom_result_table(st.session_state.get("custom_run_results", {}))


def _render_question_runner(question: Dict[str, Any], question_index: int, selected_variants: list[str], top_k: int, two_step: bool) -> None:
    source = st.segmented_control(
        "Nguồn câu hỏi",
        ["Test set", "Câu hỏi tuỳ ý"],
        default="Test set",
        required=True,
        key="runner_question_source",
        persist_state="session",
    )
    if source == "Câu hỏi tuỳ ý":
        _render_custom_question_runner(selected_variants, top_k, two_step)
    else:
        _render_test_set_runner(question, question_index, selected_variants, top_k, two_step)


def _render_trace(question_id: str, default_variant: str) -> None:
    custom_source = st.session_state.get("runner_question_source") == "Câu hỏi tuỳ ý"
    trace_subject = "câu hỏi tuỳ ý vừa chạy" if custom_source else question_id
    st.subheader(f"Agent trace · {trace_subject}")
    variant = st.selectbox("Variant để xem trace", VARIANTS, index=VARIANTS.index(default_variant), key="trace_variant")
    if custom_source:
        result = st.session_state.get("custom_run_results", {}).get(variant)
    else:
        result = select_question_result(variant, question_id, RESULTS_ROOT, SINGLE_ROOT)
    if result is None:
        if custom_source:
            st.info("Chưa có trace cho variant này. Hãy chạy variant trong Question Runner với nguồn câu hỏi tuỳ ý.")
        else:
            st.info("Chưa có artifact cho question/variant này. Hãy chạy trong Question Runner hoặc kiểm tra full results.")
        return

    _render_evidence_ribbon(result)
    metrics = st.columns(4)
    metrics[0].metric("Answer", result.get("predicted_answer") or "—")
    metrics[1].metric("Confidence", f"{float(result.get('confidence', 0.0)):.2f}")
    metrics[2].metric("Latency", f"{float(result.get('latency_seconds', 0.0)):.2f}s")
    metrics[3].metric("Total tokens", f"{int(result.get('total_tokens', 0) or 0):,}")

    metadata = result.get("metadata") or {}
    rag, plan, examiner, evaluator, raw = st.tabs(["RAG", "Plan", "Examiner", "Evaluator", "Raw JSON"])
    with rag:
        st.code(metadata.get("guidelines_preview") or "Không có RAG context trong artifact này.", language="text")
        st.json(metadata.get("rag_trace") or {})
    with plan:
        st.json(metadata.get("planner_trace") or {})
    with examiner:
        st.code(result.get("reasoning") or "Không có reasoning trace.", language="text")
        st.json(metadata.get("examiner_trace") or {})
    with evaluator:
        st.json(metadata.get("evaluator_trace") or {})
    with raw:
        st.json(result)

    if not custom_source:
        log_path = LOG_ROOT / variant / f"{question_id}.log"
        if log_path.exists():
            with st.expander("CLI log"):
                st.code(log_path.read_text(encoding="utf-8"), language="text")


def main() -> None:
    st.set_page_config(page_title="MedQA Evidence Board", page_icon="🩺", layout="wide")
    st.markdown("""
    <style>
      .stApp { background: #F4F7F8; color: #102A43; }
      h1, h2, h3 { font-family: Georgia, serif; color: #102A43; }
      [data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #D9E2EC; border-radius: 8px; padding: 0.75rem; }
      .evidence-ribbon { display: flex; gap: 0.45rem; margin: 0.8rem 0 1.2rem; flex-wrap: wrap; }
      .ribbon-stage { border-radius: 999px; padding: 0.35rem 0.8rem; font: 600 0.78rem ui-monospace, SFMono-Regular, Menlo, monospace; }
      .ribbon-stage.active.retrieval { background: #CCFBF1; color: #115E59; }
      .ribbon-stage.active.planner { background: #DBEAFE; color: #1D4ED8; }
      .ribbon-stage.active.examiner { background: #E0E7FF; color: #4338CA; }
      .ribbon-stage.active.evaluator { background: #FEF3C7; color: #92400E; }
      .ribbon-stage.muted { background: #E5E7EB; color: #6B7280; }
    </style>
    """, unsafe_allow_html=True)
    st.title("MedQA Evidence Board")
    st.caption("Ablation evidence from direct LLM to multi-agent verification")
    st.session_state.setdefault("last_test_run", None)
    st.session_state.setdefault("custom_run_results", {})

    with st.sidebar:
        st.header("Question controls")
        data_path = st.text_input("Dataset", value=MedQALoader.DEFAULT_TEST_PATH)
        try:
            questions = _load_questions(data_path)
        except Exception as error:
            st.error(f"Không thể đọc dataset: {error}")
            return
        if not questions:
            st.error("Dataset không chứa câu hỏi hợp lệ.")
            return
        question_index = st.number_input("Question index (zero-based)", min_value=0, max_value=len(questions) - 1, value=10, step=1)
        selected_variants = st.multiselect("Variant cần chạy", VARIANTS, default=["V3"])
        top_k = st.slider("RAG top-k", min_value=1, max_value=10, value=5)
        two_step = st.toggle("Two-step retrieval", value=False)
        st.caption("Chạy nhiều variant theo thứ tự đã chọn để tránh burst API.")

    question = questions[int(question_index)]
    results = _load_results()
    dashboard, runner, trace = st.tabs(["Benchmark dashboard", "Question runner", "Agent trace"])
    with dashboard:
        _render_dashboard(results)
    with runner:
        _render_question_runner(question, int(question_index), selected_variants, top_k, two_step)
    with trace:
        _render_trace(question["question_id"], selected_variants[0] if selected_variants else "V3")


if __name__ == "__main__":
    main()
