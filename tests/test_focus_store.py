import pytest

from clawmarks import config
from clawmarks.focus_store import (
    FocusConflict,
    FocusIntegrityError,
    FocusNotFound,
    FocusStore,
    FocusValidationError,
    Scope,
)


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / "state"


@pytest.fixture
def real_dir(tmp_path):
    path = tmp_path / "real"
    path.mkdir()
    (path / "real.jpg").write_bytes(b"real")
    return path


@pytest.fixture
def scope():
    return Scope("demo", "round1")


@pytest.fixture
def leg_dir(tmp_path, monkeypatch, scope):
    expeditions_dir = tmp_path / "state" / "expeditions"
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", expeditions_dir)
    path = config.leg_dir(scope.expedition, scope.leg)
    path.mkdir(parents=True)
    return path


@pytest.fixture
def store(state_dir, real_dir):
    return FocusStore(state_dir, real_dir)


@pytest.fixture
def manifest(leg_dir):
    records = []
    for tag in ("a", "b"):
        image = leg_dir / f"{tag}.png"
        image.write_bytes(tag.encode())
        records.append({"tag": tag, "file": str(image)})
    return records


def map_payload(member_tags, real_anchor_tags=None):
    return {
        "label": "Ink anchor",
        "source": {
            "view": "map",
            "kind": "map_members",
            "member_tags": member_tags,
            "real_anchor_tags": real_anchor_tags or ["real.jpg"],
            "projection_hint": {
                "projection_version": "sha256:abc",
                "polygon": [[0.1, 0.2]],
            },
        },
        "question": "  Keep these spaces  ",
        "observation": "Six clusters.",
        "hypothesis_text": "Marks survive.",
        "test_contract": None,
    }


def frontier_payload(faith, novelty, adjacent, real_anchor_tags=None, coverage_hint=None):
    payload = {
        "label": "Empty bin",
        "source": {
            "view": "coverage",
            "kind": "coverage_frontier",
            "score_ranges": {
                "faithfulness": list(faith),
                "novelty": list(novelty),
            },
            "adjacent_member_tags": list(adjacent),
            "real_anchor_tags": real_anchor_tags or ["real.jpg"],
        },
        "question": "  Push here  ",
        "observation": "Frontier cell.",
        "hypothesis_text": "Marks survive.",
        "test_contract": None,
    }
    if coverage_hint is not None:
        payload["source"]["coverage_hint"] = coverage_hint
    return payload


def test_create_map_focus_preserves_text_and_deduplicates_tags(store, scope, manifest):
    focus = store.create(
        scope,
        map_payload(["a", "a", "b"]),
        manifest,
    )

    assert focus["revision"] == 1
    assert focus["source"]["member_tags"] == ["a", "b"]
    assert focus["question"] == "  Keep these spaces  "
    assert focus["status"] == "open"
    assert "." not in focus["created_at"].split("+", 1)[0]
    assert store.get(scope, focus["focus_id"]) == focus


def test_create_rejects_cross_leg_or_duplicate_manifest_tag(store, scope, manifest):
    with pytest.raises(FocusValidationError, match="resolve exactly once"):
        store.create(scope, map_payload(["a"]), [manifest[0], manifest[0]])


def test_create_rejects_manifest_file_outside_scoped_leg(store, scope, manifest, tmp_path):
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")

    with pytest.raises(FocusIntegrityError, match="outside"):
        store.create(scope, map_payload(["a"]), [{"tag": "a", "file": str(outside)}])


def test_create_rejects_unknown_real_anchor(store, scope, manifest):
    with pytest.raises(FocusValidationError, match="real anchor"):
        store.create(scope, map_payload(["a"], ["missing.jpg"]), manifest)


def test_stale_update_returns_current_record(store, scope, manifest):
    focus = store.create(scope, map_payload(["a"]), manifest)
    current = store.update(scope, focus["focus_id"], 1, {"label": "new"})

    with pytest.raises(FocusConflict) as exc:
        store.update(scope, focus["focus_id"], 1, {"label": "stale"})

    assert exc.value.current == current


def test_update_rejects_invalid_id_and_unsupported_key(store, scope, manifest):
    with pytest.raises(FocusValidationError):
        store.get(scope, "focus_not-a-uuid")

    focus = store.create(scope, map_payload(["a"]), manifest)
    with pytest.raises(FocusValidationError, match="unsupported"):
        store.update(scope, focus["focus_id"], 1, {"source": {}})


def test_archived_focus_rejects_update(store, scope, manifest):
    focus = store.create(scope, map_payload(["a"]), manifest)
    archived = store.archive(scope, focus["focus_id"], 1)

    with pytest.raises(FocusValidationError, match="archived"):
        store.update(scope, focus["focus_id"], archived["revision"], {"label": "new"})


def test_list_status_filtering(store, scope, manifest):
    open_focus = store.create(scope, map_payload(["a"]), manifest)
    archived_focus = store.create(scope, map_payload(["b"]), manifest)
    store.archive(scope, archived_focus["focus_id"], 1)

    assert [item["focus_id"] for item in store.list(scope)] == [
        archived_focus["focus_id"],
        open_focus["focus_id"],
    ]
    assert [item["focus_id"] for item in store.list(scope, status="open")] == [
        open_focus["focus_id"]
    ]
    assert [item["focus_id"] for item in store.list(scope, status="archived")] == [
        archived_focus["focus_id"]
    ]


def test_malformed_json_is_preserved_and_reports_integrity_error(store, scope, manifest):
    focus = store.create(scope, map_payload(["a"]), manifest)
    path = store.state_dir / "foci" / scope.expedition / scope.leg / f"{focus['focus_id']}.json"
    original = b"{not valid json"
    path.write_bytes(original)

    with pytest.raises(FocusIntegrityError, match="JSON") as exc:
        store.update(scope, focus["focus_id"], 1, {"label": "new"})

    assert exc.value.path == path
    assert path.read_bytes() == original


def test_missing_focus_raises_not_found(store, scope):
    with pytest.raises(FocusNotFound):
        store.get(scope, "focus_0123456789abcdef0123456789abcdef")


def test_create_frontier_focus_requires_empty_adjacent_cell(store, scope, manifest):
    focus = store.create(
        scope,
        frontier_payload(faith=[-0.2, 0.1], novelty=[0.8, 1.1], adjacent=["a"]),
        manifest,
        coverage_cells=[
            {
                "faith_lo": -0.2,
                "faith_hi": 0.1,
                "novelty_lo": 0.8,
                "novelty_hi": 1.1,
                "count": 0,
                "frontier": True,
            }
        ],
    )

    assert focus["source"]["kind"] == "coverage_frontier"


@pytest.mark.parametrize(
    "faith,novelty",
    [
        ([0.2, 0.2], [0.1, 0.2]),
        ([-1.1, 0], [0.1, 0.2]),
        ([0, 1], [1.8, 2.1]),
    ],
)
def test_frontier_ranges_must_be_ordered_and_in_domain(
    store, scope, manifest, faith, novelty
):
    with pytest.raises(FocusValidationError):
        store.create(
            scope,
            frontier_payload(faith, novelty, ["a"]),
            manifest,
            coverage_cells=[],
        )


@pytest.mark.parametrize(
    "cell",
    [
        None,
        {
            "faith_lo": -0.2,
            "faith_hi": 0.1,
            "novelty_lo": 0.8,
            "novelty_hi": 1.1,
            "count": 1,
            "frontier": True,
        },
        {
            "faith_lo": -0.2,
            "faith_hi": 0.1,
            "novelty_lo": 0.8,
            "novelty_hi": 1.1,
            "count": 0,
            "frontier": False,
        },
    ],
)
def test_create_frontier_rejects_invalid_coverage_cells(
    store, scope, manifest, cell
):
    cells = [] if cell is None else [cell]
    with pytest.raises(FocusValidationError):
        store.create(
            scope,
            frontier_payload(faith=[-0.2, 0.1], novelty=[0.8, 1.1], adjacent=["a"]),
            manifest,
            coverage_cells=cells,
        )


def test_create_frontier_focus_preserves_coverage_hint_and_normalizes_ranges(
    store, scope, manifest
):
    hint = {
        "binning_version": "sha256:abc",
        "metric_domains": {"faithfulness": [-1.0, 1.0], "novelty": [0.0, 2.0]},
        "row": 4,
        "column": 3,
    }
    faith, novelty = [-0.2, 0.1], [0.8, 1.1]
    cells = [
        {
            "faith_lo": -0.2,
            "faith_hi": 0.1,
            "novelty_lo": 0.8,
            "novelty_hi": 1.1,
            "count": 0,
            "frontier": True,
        }
    ]

    focus = store.create(
        scope,
        frontier_payload(faith, novelty, ["a"], coverage_hint=hint),
        manifest,
        coverage_cells=cells,
    )

    assert focus["source"]["score_ranges"] == {
        "faithfulness": [float(v) for v in faith],
        "novelty": [float(v) for v in novelty],
    }
    assert focus["source"]["adjacent_member_tags"] == ["a"]
    assert focus["source"]["coverage_hint"] == hint
