"""Deployment entry point for the IdeaGrade Streamlit application."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="IdeaGrade · Explainable grading",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from analysis_service import AnalysisBundle, analyze_answers, relationship_rows
from data_store import load_teacher_scores, save_teacher_score
from gnn_model import load_trained_model
from graph_builder import build_idea_graph
from settings import APP_VERSION
from utils import draw_graph


EXAMPLE_REFERENCE = (
    "Plants use sunlight to produce glucose. "
    "Plants release oxygen during photosynthesis."
)
EXAMPLE_STUDENT = (
    "Plants use sunlight to produce glucose. "
    "Plants release carbon dioxide during photosynthesis."
)


@st.cache_resource(show_spinner=False)
def get_model():
    return load_trained_model()


@st.cache_data(ttl="30s", max_entries=1, show_spinner=False)
def get_training_count() -> int:
    return len(load_teacher_scores())


def load_example() -> None:
    st.session_state.reference_answer = EXAMPLE_REFERENCE
    st.session_state.student_answer = EXAMPLE_STUDENT
    st.session_state.analysis = None


def score_band(score: float) -> tuple[str, str, str]:
    if score >= 85:
        return "Strong structural match", "success", ":material/check_circle:"
    if score >= 60:
        return "Partial structural match", "warning", ":material/rule:"
    return "Teacher review recommended", "error", ":material/rate_review:"


def render_claims(title: str, claims: tuple[str, ...], kind: str) -> None:
    with st.container(border=True, height="stretch"):
        st.subheader(title)
        if not claims:
            st.success("No issues detected.", icon=":material/check_circle:")
            return
        for claim in claims:
            if kind == "missing":
                st.error(claim, icon=":material/remove_circle:")
            else:
                st.warning(claim, icon=":material/add_circle:")


st.session_state.setdefault("reference_answer", "")
st.session_state.setdefault("student_answer", "")
st.session_state.setdefault("analysis", None)

st.title(":material/hub: IdeaGrade")
st.caption(
    "Explainable, graph-aware grading for short factual answers. "
    "Compare concepts, relationships, and reasoning structure in seconds."
)
st.markdown(
    ":violet-badge[:material/account_tree: Idea graphs] "
    ":blue-badge[:material/psychology: GNN assisted] "
    ":green-badge[:material/visibility: Explainable]"
)

model = get_model()
training_count = get_training_count()

with st.sidebar:
    st.subheader(":material/model_training: Model status")
    if model is not None:
        st.badge(
            "Trained model ready",
            icon=":material/check_circle:",
            color="green",
        )
        st.metric(
            "Validation MAE",
            f"{getattr(model, 'validation_mae', 0.0):.2f} points",
            border=True,
            help="Average absolute error on topics excluded from training.",
        )
    else:
        st.badge("Baseline only", icon=":material/info:", color="orange")

    st.metric("Available examples", training_count, border=True)
    st.button(
        "Load example",
        icon=":material/science:",
        on_click=load_example,
        width="stretch",
    )

    with st.expander("Scoring rubric", icon=":material/analytics:"):
        st.markdown(
            "- **30%** concept coverage\n"
            "- **45%** relationship accuracy\n"
            "- **25%** argument structure"
        )
        st.caption(
            "The GNN adds a learned estimate from teacher-scored comparison graphs."
        )

    st.caption(
        f"IdeaGrade v{APP_VERSION} · Teacher-assist only. "
        "The final grading decision remains with the educator."
    )

with st.container(border=True):
    st.subheader(":material/compare_arrows: Compare answers")
    st.caption(
        "Use complete sentences for the clearest concept and relationship extraction."
    )
    with st.form("answer_comparison", border=False):
        reference_column, student_column = st.columns(2, gap="large")
        with reference_column:
            st.text_area(
                "Teacher reference answer",
                key="reference_answer",
                height=220,
                placeholder="Enter the ideal answer or marking-scheme explanation…",
            )
        with student_column:
            st.text_area(
                "Student answer",
                key="student_answer",
                height=220,
                placeholder="Paste the student's answer here…",
            )
        submitted = st.form_submit_button(
            "Analyze answer",
            type="primary",
            icon=":material/analytics:",
            width="stretch",
        )

if submitted:
    reference = st.session_state.reference_answer.strip()
    student = st.session_state.student_answer.strip()
    if not reference or not student:
        st.error(
            "Add both a teacher reference and a student answer before analyzing.",
            icon=":material/error:",
        )
    else:
        try:
            with st.spinner("Building and comparing idea graphs…"):
                st.session_state.analysis = analyze_answers(reference, student, model)
        except RuntimeError as error:
            st.session_state.analysis = None
            st.error(str(error), icon=":material/error:")

analysis: AnalysisBundle | None = st.session_state.analysis
if analysis is None:
    st.info(
        "Enter two answers above or load the example to begin.",
        icon=":material/lightbulb:",
    )
    st.stop()

grade = analysis.grade
band_label, band_kind, band_icon = score_band(grade.score)
getattr(st, band_kind)(
    f"**{band_label} · {grade.score:.1f}/100**",
    icon=band_icon,
)

metric_columns = st.columns(4, gap="medium")
metric_columns[0].metric("Structural score", f"{grade.score:.1f}/100", border=True)
metric_columns[1].metric("Concept coverage", f"{grade.concept_score:.1f}%", border=True)
metric_columns[2].metric(
    "Relationship accuracy",
    f"{grade.relationship_score:.1f}%",
    border=True,
)
metric_columns[3].metric(
    "Argument structure",
    f"{grade.structure_score:.1f}%",
    border=True,
)

if analysis.gnn_score is not None:
    difference = analysis.gnn_score - grade.score
    model_columns = st.columns([1, 1, 2], vertical_alignment="center")
    model_columns[0].metric(
        "GNN prediction",
        f"{analysis.gnn_score:.1f}/100",
        delta=f"{difference:+.1f} vs structural",
        delta_color="off",
        border=True,
    )
    model_columns[1].metric(
        "Model uncertainty",
        f"±{analysis.validation_mae:.2f}",
        border=True,
        help="Validation mean absolute error on unseen topics.",
    )
    with model_columns[2].container(border=True, height="stretch"):
        st.markdown("**How to read this**")
        st.caption(
            "Use the structural score for direct evidence and the GNN prediction "
            "as a learned second opinion. Review disagreements before grading."
        )

summary_tab, evidence_tab, graph_tab = st.tabs(
    [
        ":material/summarize: Summary",
        ":material/fact_check: Evidence",
        ":material/account_tree: Idea graphs",
    ],
    key="result_view",
    on_change="rerun",
)

if summary_tab.open:
    with summary_tab:
        missing_column, extra_column = st.columns(2, gap="large")
        with missing_column:
            render_claims(
                "Missing reference relationships",
                grade.missing_relationships,
                "missing",
            )
        with extra_column:
            render_claims(
                "Additional student relationships",
                grade.extra_relationships,
                "extra",
            )

        with st.container(border=True):
            st.subheader(":material/rate_review: Teacher takeaway")
            if grade.score >= 85:
                st.write(
                    "The student preserves most concepts and structural relationships. "
                    "Check wording and subject-specific detail before confirming the mark."
                )
            elif grade.score >= 60:
                st.write(
                    "The core reasoning is partially present, but one or more relationships "
                    "need correction or elaboration."
                )
            else:
                st.write(
                    "The answer diverges substantially from the reference structure. "
                    "Review the extracted evidence and provide targeted feedback."
                )

if evidence_tab.open:
    with evidence_tab:
        reference_table, student_table = st.columns(2, gap="large")
        with reference_table:
            st.subheader("Reference relationships")
            st.dataframe(
                relationship_rows(analysis.reference),
                width="stretch",
                hide_index=True,
            )
        with student_table:
            st.subheader("Student relationships")
            st.dataframe(
                relationship_rows(analysis.student),
                width="stretch",
                hide_index=True,
            )

        with st.container(horizontal=True, horizontal_alignment="left"):
            st.download_button(
                "Download report",
                data=grade.to_report(),
                file_name="ideagrade_report.txt",
                mime="text/plain",
                icon=":material/download:",
            )

        with st.expander(
            "Add teacher calibration",
            icon=":material/model_training:",
        ):
            st.caption(
                "Save the educator's final score as supervised data for future retraining."
            )
            with st.form("teacher_calibration", border=False):
                teacher_score = st.number_input(
                    "Teacher's final score",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(grade.score),
                    step=1.0,
                )
                calibration_saved = st.form_submit_button(
                    "Save calibration",
                    icon=":material/save:",
                    width="stretch",
                )
            if calibration_saved:
                save_teacher_score(
                    analysis.reference,
                    analysis.student,
                    teacher_score,
                )
                get_training_count.clear()
                st.toast(
                    "Teacher calibration saved.",
                    icon=":material/check_circle:",
                )

if graph_tab.open:
    with graph_tab:
        st.caption(
            "Nodes represent concepts; directed labels represent extracted relationships."
        )
        dark_mode = st.context.theme.type == "dark"
        reference_graph, student_graph = st.columns(2, gap="large")
        with reference_graph:
            with st.container(border=True):
                st.subheader("Reference idea graph")
                st.pyplot(
                    draw_graph(
                        build_idea_graph(analysis.reference),
                        "Reference answer",
                        dark=dark_mode,
                    ),
                    clear_figure=True,
                )
                st.caption(
                    "\n".join(analysis.reference_edges)
                    if analysis.reference_edges
                    else "No explicit relationships found."
                )
        with student_graph:
            with st.container(border=True):
                st.subheader("Student idea graph")
                st.pyplot(
                    draw_graph(
                        build_idea_graph(analysis.student),
                        "Student answer",
                        dark=dark_mode,
                    ),
                    clear_figure=True,
                )
                st.caption(
                    "\n".join(analysis.student_edges)
                    if analysis.student_edges
                    else "No explicit relationships found."
                )
