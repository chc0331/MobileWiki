"""Local, artifact-only clustering explorer. Run with Streamlit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from visualize import (
    BASE_DIR, bundle_display_name, discover_bundles, error_pairs, group_summaries, load_bundle,
    recluster_development, same_dataset, selected_doc_ids,
)


@st.cache_data(show_spinner=False)
def cached_bundle(path: str, modified_ns: int) -> dict:
    return load_bundle(Path(path))


@st.cache_data(show_spinner=False)
def cached_errors(documents: list[dict], groups: dict[str, int]) -> list[dict]:
    return error_pairs(documents, groups)


@st.cache_data(show_spinner=False)
def cached_distances(documents: list[dict], vectors: np.ndarray, config: dict) -> np.ndarray:
    from clustering import ClusterConfig, distance_matrix

    return distance_matrix(documents, vectors, ClusterConfig.from_dict(config))


def bundle_title(path: Path) -> str:
    import json

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return bundle_display_name(manifest)


def view_for(bundle: dict, split: str) -> dict:
    key = f"exploration:{bundle['root']}:{split}"
    return st.session_state.get(key, bundle["splits"][split])


def changed_selection(key: str, values: list[str]) -> bool:
    previous = st.session_state.get(key, [])
    st.session_state[key] = values
    return values != previous


def show_metrics(bundle: dict, split: str, view: dict) -> None:
    from evaluate import all_event_pair_f1

    metrics = all_event_pair_f1(view["documents"], view["groups"])
    columns = st.columns(5)
    for column, (label, value) in zip(columns, (
        ("전체 사건 쌍 F1", f"{metrics['f1']:.3f}"),
        ("정밀도", f"{metrics['precision']:.3f}"),
        ("재현율", f"{metrics['recall']:.3f}"),
        ("오결합", f"{metrics['fp']:,}쌍"),
        ("놓친 관련 쌍", f"{metrics['fn']:,}쌍"),
    )):
        column.metric(label, value)
    count = len(set(view["groups"].values()))
    status = "개발 탐색 설정" if view.get("exploratory") else "저장된 평가 설정"
    st.caption(f"{status} · {'개발' if split == 'dev' else '시험'} 전체 {len(view['documents']):,}개 메모 · {count:,}개 묶음 · 필터와 관계없이 전체 split을 계산한 지표")
    if split == "test":
        search = bundle["evaluation"].get("test_dense_search", {})
        if search:
            st.caption(f"저장된 검색 Recall@5 {search['recall_at_5']:.1%} ({search['hits_at_5']}/{search['total']}) · 군집 설정 탐색으로 검색 결과는 바뀌지 않습니다.")


def note_detail(document: dict, groups: dict[str, int]) -> None:
    st.markdown(f"**{document['title']}**")
    st.caption(f"{document['doc_id']} · 예측 묶음 {groups[document['doc_id']]} · 실제 사건 {document.get('event_id', '정보 없음')}")
    st.write(document["text"])
    from clustering import text_date

    extracted = text_date(document)
    st.caption(
        f"본문에서 확정한 날짜: {extracted if extracted is not None else '정보 없음'} · "
        f"정답 메타데이터 날짜: {document.get('date', '정보 없음')} · 주제: {document.get('category', '정보 없음')}"
    )


def scatter(view: dict, bundle: dict, split: str, key: str, highlighted: list[str] | None = None) -> list[str]:
    documents = view["documents"]
    groups = view["groups"]
    controls = st.columns(3)
    method = controls[0].selectbox("2차원 보기", ("PCA", "t-SNE"), key=f"projection:{key}")
    color = controls[1].selectbox("색상 기준", ("예측 묶음", "주제", "실제 사건"), key=f"color:{key}")
    categories = sorted({doc.get("category", "정보 없음") for doc in documents})
    category = controls[2].selectbox("주제", ["전체", *categories], key=f"category:{key}")
    coordinates = view["pca" if method == "PCA" else "tsne"]
    table = pd.DataFrame({
        "doc_id": [doc["doc_id"] for doc in documents],
        "title": [doc["title"] for doc in documents],
        "예측 묶음": [str(groups[doc["doc_id"]]) for doc in documents],
        "주제": [doc.get("category", "정보 없음") for doc in documents],
        "실제 사건": [doc.get("event_id", "정보 없음") for doc in documents],
        "x": coordinates[:, 0], "y": coordinates[:, 1],
    })
    if category != "전체":
        table = table[table["주제"] == category]
    fig = px.scatter(
        table, x="x", y="y", color=color, hover_name="title", custom_data=["doc_id"],
        hover_data={"doc_id": True, "주제": True, "실제 사건": True, "x": False, "y": False},
    )
    fig.update_traces(marker={"size": 7, "opacity": 0.75})
    if highlighted:
        chosen = table[table["doc_id"].isin(highlighted)]
        fig.add_trace(go.Scatter(
            x=chosen["x"], y=chosen["y"], mode="markers", name="선택한 메모",
            marker={"size": 15, "symbol": "circle-open", "line": {"width": 3}},
            customdata=[[doc_id] for doc_id in chosen["doc_id"]],
            hovertext=chosen["title"], hoverinfo="text",
        ))
    fig.update_layout(
        height=530, margin={"l": 0, "r": 0, "t": 15, "b": 0},
        clickmode="event+select", dragmode="pan",
        showlegend=table[color].nunique() <= 20, xaxis_title="축 1", yaxis_title="축 2",
    )
    selection = st.plotly_chart(
        fig, width="stretch", key=f"plot:{key}", on_select="rerun",
        selection_mode=("points", "box", "lasso"),
        alt="메모의 임베딩을 2차원으로 투영한 군집 지도. 점을 선택하면 원문을 확인할 수 있습니다.",
    )
    st.caption("2차원 좌표는 탐색용입니다. 군집 지표와 거리 히트맵은 원래 임베딩 공간에서 계산합니다.")
    return selected_doc_ids(selection, set(table["doc_id"]))


def map_tab(bundle: dict, split: str, view: dict) -> None:
    key = f"map:{bundle['root']}:{split}"
    selected = scatter(view, bundle, split, key)
    selection_changed = changed_selection(f"last-selection:{key}", selected)
    summaries = group_summaries(view["documents"], view["groups"], view["labels"])
    by_id = {doc["doc_id"]: doc for doc in view["documents"]}
    if selected:
        if selection_changed:
            st.session_state[f"group:{key}"] = view["groups"][selected[0]]
        with st.expander(f"지도에서 선택한 메모 {len(selected)}개", expanded=True):
            for doc_id in selected[:8]:
                note_detail(by_id[doc_id], view["groups"])
            if len(selected) > 8:
                st.caption("선택 메모의 처음 8개를 표시합니다. 묶음 목록에서 나머지 원문을 확인할 수 있습니다.")
    summary_by_id = {row["group_id"]: row for row in summaries}
    if st.session_state.get(f"group:{key}") not in summary_by_id:
        st.session_state.pop(f"group:{key}", None)
    group_id = st.selectbox(
        "묶음 선택 · 오결합이 많은 순서", [row["group_id"] for row in summaries], key=f"group:{key}",
        format_func=lambda value: f"{value} · {summary_by_id[value]['label']} · {summary_by_id[value]['documents']}개 메모 / {summary_by_id[value]['events']}개 사건 / 오결합 {summary_by_id[value]['false_merge_pairs']}쌍",
    )
    members = [doc for doc in view["documents"] if view["groups"][doc["doc_id"]] == group_id]
    labels = next((label for label in view["labels"] if int(label["group_id"]) == group_id), {})
    st.caption(f"태그 선택 단계: {labels.get('label_stage', '기존 규칙')} · 출처: {', '.join(labels.get('source_doc_ids', [])) or '없음'}")
    member_table = pd.DataFrame([
        {"문서": doc["doc_id"], "제목": doc["title"], "실제 사건": doc.get("event_id", "정보 없음"), "날짜(정답 메타데이터)": doc.get("date", "정보 없음"), "본문": doc["text"]}
        for doc in members
    ])
    st.dataframe(member_table, hide_index=True, width="stretch")
    if len(members) > 1:
        display = sorted(members, key=lambda doc: (doc.get("event_id", ""), doc["doc_id"]))[:60]
        indices = [next(index for index, doc in enumerate(view["documents"]) if doc["doc_id"] == member["doc_id"]) for member in display]
        metric = st.radio("거리 히트맵", ("실제 군집 거리", "원본 코사인 거리"), horizontal=True, key=f"distance:{key}")
        if metric == "원본 코사인 거리":
            vectors = view["raw"][indices]
            distances = np.clip(1 - vectors @ vectors.T, 0, 2)
        else:
            profile = view.get("profile", bundle["profile"])
            full = cached_distances(view["documents"], view["corrected"], profile["config"])
            distances = full[np.ix_(indices, indices)]
        fig = px.imshow(distances, x=[doc["doc_id"] for doc in display], y=[doc["doc_id"] for doc in display], labels={"color": "거리"}, color_continuous_scale="Viridis", aspect="auto")
        fig.update_layout(height=450)
        st.plotly_chart(fig, width="stretch", alt="선택한 묶음 안에서 실제 사건 순으로 정렬한 문서 간 거리 히트맵")
        if len(members) > 60:
            st.caption("큰 묶음의 히트맵은 실제 사건 순으로 정렬한 처음 60개 문서입니다.")


def errors_tab(bundle: dict, split: str, view: dict) -> None:
    rows = cached_errors(view["documents"], view["groups"])
    if not rows:
        st.info("오결합이나 과분할 쌍이 없습니다.")
        return
    kind = st.radio("오류 종류", ("전체", "오결합", "과분할"), horizontal=True, key=f"error-kind:{bundle['root']}:{split}")
    feature = st.selectbox("오류 조건", ("전체", "같은 날짜·주제", "같은 가족", "같은 정답값", "답 없는 메모끼리"), key=f"error-feature:{bundle['root']}:{split}")
    filtered = [row for row in rows if kind == "전체" or row["error"] == kind]
    checks = {
        "같은 날짜·주제": lambda row: row["same_date"] is True and row["same_category"] is True,
        "같은 가족": lambda row: row["same_family"] is True,
        "같은 정답값": lambda row: row["same_fact"] is True,
        "답 없는 메모끼리": lambda row: row["both_answerless"] is True,
    }
    if feature in checks:
        filtered = [row for row in filtered if checks[feature](row)]
    st.caption(f"조건에 맞는 오류 {len(filtered):,}쌍 · 조건 필터는 합성 정답 메타데이터를 사용한 진단이며 군집 입력으로 사용하지 않습니다.")
    if not filtered:
        return
    pages = max(1, (len(filtered) + 99) // 100)
    page = st.number_input("오류 페이지", 1, pages, 1, key=f"error-page:{bundle['root']}:{split}:{kind}:{feature}")
    current = filtered[(page - 1) * 100:page * 100]
    st.dataframe(pd.DataFrame(current).rename(columns={"error": "오류", "left_id": "왼쪽 문서", "right_id": "오른쪽 문서"}), hide_index=True, width="stretch")
    chosen = st.selectbox("원문 대조", range(len(current)), format_func=lambda index: f"{current[index]['error']} · {current[index]['left_id']} ↔ {current[index]['right_id']}", key=f"error-pair:{bundle['root']}:{split}:{kind}:{feature}:{page}")
    by_id = {doc["doc_id"]: doc for doc in view["documents"]}
    left, right = st.columns(2)
    with left:
        note_detail(by_id[current[chosen]["left_id"]], view["groups"])
    with right:
        note_detail(by_id[current[chosen]["right_id"]], view["groups"])


def compare_tab(bundle: dict, split: str, catalog: list[Path]) -> None:
    choices = [path for path in catalog if path.resolve() != bundle["root"]]
    if not choices:
        st.info("다른 모델 평가를 export하면 나란히 비교할 수 있습니다.")
        return
    path = st.selectbox("비교할 평가 실행", choices, format_func=bundle_title)
    other = cached_bundle(str(path), (path / "manifest.json").stat().st_mtime_ns)
    if split not in other["splits"]:
        st.info("비교 실행에 선택한 split이 없습니다.")
        return
    compatible = same_dataset(bundle["manifest"], other["manifest"])
    st.caption("같은 데이터에서만 문서 선택을 함께 강조합니다. 모델별 투영 축은 달라 지도상의 절대 위치 이동은 성능 변화로 해석하지 않습니다.")
    if not compatible:
        st.info("서로 다른 데이터 revision입니다. 지표와 지도를 각각 비교하며 같은 문서 ID를 연결하지 않습니다.")
    selection_key = f"comparison-selection:{bundle['manifest']['dataset_documents_sha256']}:{split}"
    highlighted = st.session_state.get(selection_key, []) if compatible else []
    left, right = st.columns(2)
    for column, current in ((left, bundle), (right, other)):
        with column:
            st.markdown(f"**{bundle_display_name(current['manifest'])}**")
            current_view = current["splits"][split]
            show_metrics(current, split, current_view)
            plot_key = f"compare:{current['root']}:{split}"
            selected = scatter(current_view, current, split, plot_key, highlighted)
            is_new = changed_selection(f"last-selection:{plot_key}", selected)
            if compatible and is_new and selected and selected != highlighted:
                st.session_state[selection_key] = selected
                st.rerun()
    if compatible and highlighted:
        records = []
        for doc_id in highlighted[:20]:
            for current in (bundle, other):
                view = current["splits"][split]
                group = view["groups"][doc_id]
                members = sorted(candidate for candidate, value in view["groups"].items() if value == group)
                records.append({"메모": doc_id, "실행": bundle_display_name(current["manifest"]), "묶음 문서 수": len(members), "포함 문서": ", ".join(members)})
        st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")


def exploration_tab(bundle: dict, split: str) -> None:
    if split != "dev":
        st.info("군집 설정 변경은 개발 자료에서 제공합니다. 왼쪽 split을 ‘개발’로 바꾸면 실험할 수 있습니다.")
        st.json(bundle["profile"]["config"], expanded=False)
        return
    from clustering import ClusterConfig

    settings = bundle["profile"]["config"]
    prefix = f"controls:{bundle['root']}"
    with st.form(f"experiment:{bundle['root']}"):
        left, right = st.columns(2)
        mode = left.selectbox("공통 배경 방향 보정", ("none", "center", "abtt"), index=("none", "center", "abtt").index(settings.get("postprocess", "none")),
                              format_func=lambda value: {"none": "보정 안 함", "center": "평균 중심화", "abtt": "공통 방향 제거(ABTT)"}[value])
        components = left.selectbox("제거할 상위 주성분 수", (1, 3), index=1 if settings.get("remove_components") == 3 else 0)
        linkage = left.selectbox("군집 연결", ("average", "complete"), index=int(settings.get("linkage") == "complete"),
                                 format_func=lambda value: {"average": "평균 연결(average)", "complete": "엄격 연결(complete)"}[value])
        threshold = left.number_input("군집 거리 기준", min_value=0.01, max_value=2.0, value=float(settings["threshold"]), step=0.01)
        window = right.selectbox("날짜 간격 기준(일)", (7, 30), index=int(settings.get("date_window_days", 7) == 30))
        date_penalty = right.slider("확정 날짜가 다른 메모의 거리 가산", 0.0, 0.3, float(settings.get("date_penalty", 0)), 0.05)
        mutual_k = right.selectbox("상호 근접 이웃 수 · 0은 사용 안 함", (0, 5, 10, 20), index=(0, 5, 10, 20).index(settings.get("mutual_k", 0)))
        neighbor_penalty = right.slider("상호 이웃이 아닌 연결의 거리 가산", 0.0, 0.3, float(settings.get("neighbor_penalty", 0)), 0.05)
        resplit = left.checkbox("큰 묶음·거리가 넓은 묶음 재분할", value=settings.get("resplit", False))
        maximum = right.selectbox("큰 묶음 기준", (16, 32), index=int(settings.get("max_cluster_size", 16) == 32))
        apply = st.form_submit_button("개발 자료에 적용", type="primary")
    key = f"exploration:{bundle['root']}:dev"
    if st.button("저장된 평가 설정으로 되돌리기", key=f"reset:{prefix}"):
        st.session_state.pop(key, None)
        st.rerun()
    if apply:
        config = ClusterConfig.from_dict({
            **settings, "postprocess": mode, "remove_components": components if mode == "abtt" else 0,
            "linkage": linkage, "threshold": threshold, "date_window_days": window,
            "date_penalty": date_penalty, "mutual_k": mutual_k, "neighbor_penalty": neighbor_penalty,
            "resplit": resplit, "max_cluster_size": maximum,
        })
        with st.spinner("저장된 벡터로 개발 자료를 다시 묶고 있습니다."):
            st.session_state[key] = recluster_development(bundle, split, config)
        st.rerun()
    st.caption("탐색 결과는 현재 브라우저 세션에만 남습니다. 저장된 평가 파일과 시험 결과를 덮어쓰지 않습니다.")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", type=Path, default=BASE_DIR / "artifacts")
    args, _ = parser.parse_known_args()
    st.set_page_config(page_title="메모 군집 탐색", page_icon="🔎", layout="wide")
    st.title("메모 군집 탐색")
    catalog = discover_bundles(args.root)
    if not catalog:
        st.info("아직 시각화 자료가 없습니다. 평가 산출물을 export한 뒤 이 화면을 새로고침하세요.")
        st.code(".venv/bin/python visualize.py export --dataset <dataset> --evaluation <evaluation.json> --output <visualization/run>", language="bash")
        st.stop()
    path = st.sidebar.selectbox("평가 실행", catalog, format_func=bundle_title)
    try:
        bundle = cached_bundle(str(path), (path / "manifest.json").stat().st_mtime_ns)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"시각화 자료를 읽을 수 없습니다: {error}")
        st.stop()
    splits = [name for name in ("test", "dev") if name in bundle["splits"]]
    split = st.sidebar.selectbox("자료 분할", splits, format_func=lambda value: "시험 · 고정 결과" if value == "test" else "개발 · 설정 탐색")
    st.sidebar.caption(f"데이터: {bundle['manifest']['dataset_revision']}")
    st.sidebar.caption("군집 입력: 알려진 인물 이름 가림" if bundle["evaluation"].get("clustering_deidentify") else "군집 입력: 인물 이름 유지")
    st.sidebar.caption(f"기반 모델: {bundle['manifest']['base_revision'][:12]}")
    st.sidebar.caption(f"데이터 해시: {bundle['manifest']['dataset_documents_sha256'][:12]}")
    if bundle["manifest"].get("fit_scope") == "foreign_training_dataset":
        st.sidebar.caption("임베딩 보정은 원래 학습 데이터에서 고정한 통계를 사용합니다.")
    st.caption("사람 검수 전 잠정 자료입니다." if "provisional" in bundle["manifest"]["status"] else f"검수 상태: {bundle['manifest']['status']}")
    view = view_for(bundle, split)
    show_metrics(bundle, split, view)
    map_view, errors, comparison, experimentation = st.tabs(("군집 지도", "오결합·과분할", "모델 비교", "개발 실험"))
    with map_view:
        map_tab(bundle, split, view)
    with errors:
        errors_tab(bundle, split, view)
    with comparison:
        compare_tab(bundle, split, catalog)
    with experimentation:
        exploration_tab(bundle, split)


if __name__ == "__main__":
    main()
