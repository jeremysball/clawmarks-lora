"""Durable, expedition/leg-scoped Focus records."""

from __future__ import annotations

import copy
import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from clawmarks import config
from clawmarks.atomic_io import atomic_json_write
from clawmarks.durable_records import new_id, record_locks, utc_now


_FOCUS_ID_RE = re.compile(r"focus_[0-9a-f]{32}\Z")
_ALLOWED_UPDATE_KEYS = frozenset(
    {"label", "question", "observation", "hypothesis_text", "test_contract"}
)


@dataclass(frozen=True)
class Scope:
    """The expedition and leg that own a Focus record."""

    expedition: str
    leg: str


class FocusNotFound(LookupError):
    """Raised when a requested Focus record does not exist."""

    def __init__(self, focus_id: str):
        self.focus_id = focus_id
        super().__init__(f"Focus not found: {focus_id}")


class FocusConflict(Exception):
    """Raised when a mutation uses a stale revision."""

    def __init__(self, current: dict[str, Any]):
        self.current = copy.deepcopy(current)
        super().__init__("Focus revision conflict")


class FocusIntegrityError(Exception):
    """Raised when a persisted record or evidence path cannot be trusted."""

    def __init__(self, path: Path, detail: str):
        self.path = Path(path)
        self.detail = detail
        super().__init__(f"Focus integrity error at {self.path}: {detail}")


class FocusValidationError(ValueError):
    """Raised when a Focus request violates the version-one schema."""


class FocusStore:
    """Persist map-member Focus records below one state directory."""

    def __init__(self, state_dir: Path, real_dir: Path):
        self.state_dir = Path(state_dir)
        self.real_dir = Path(real_dir)

    def list(self, scope: Scope, status: str | None = None) -> List[dict]:
        """Return the records for ``scope``, newest updates first."""
        if status not in (None, "open", "archived"):
            raise FocusValidationError(f"unsupported Focus status: {status!r}")

        directory = self._scope_dir(scope)
        if not directory.exists():
            return []
        if not directory.is_dir():
            raise FocusIntegrityError(directory, "Focus directory is not a directory")

        records: List[dict[str, Any]] = []
        for path in directory.glob("*.json"):
            try:
                focus_id = self._validate_focus_id(path.stem)
            except FocusValidationError as exc:
                raise FocusIntegrityError(path, str(exc)) from exc
            records.append(self._read_record(path, scope, focus_id))

        if status is not None:
            records = [record for record in records if record["status"] == status]

        records.sort(
            key=lambda record: (
                record["updated_at"],
                record["created_at"],
                record["status"] == "archived",
                record["focus_id"],
            ),
            reverse=True,
        )
        return records

    def get(self, scope: Scope, focus_id: str) -> dict:
        """Load one Focus from the requested scope."""
        path = self._record_path(scope, focus_id)
        return self._read_record(path, scope, focus_id)

    def create(
        self,
        scope: Scope,
        payload: dict,
        manifest: List[dict],
        coverage_cells: List[dict] | None = None,
    ) -> dict:
        """Validate and persist a new Focus (map_members or coverage_frontier)."""
        if not isinstance(payload, dict):
            raise FocusValidationError("Focus payload must be an object")
        raw_source = payload.get("source")
        if not isinstance(raw_source, dict):
            raise FocusValidationError("Focus source must be an object")
        kind = raw_source.get("kind")
        if kind == "map_members":
            del coverage_cells
            source = self._validate_map_source(scope, raw_source, manifest)
        elif kind == "coverage_frontier":
            if not isinstance(coverage_cells, list):
                raise FocusValidationError("coverage_cells must be a list")
            source = self._validate_frontier_source(
                scope, raw_source, manifest, coverage_cells
            )
        else:
            raise FocusValidationError(f"unsupported Focus source kind: {kind!r}")

        now = self._whole_second_utc()
        focus_id = self._validate_focus_id(new_id("focus"))
        path = self._record_path(scope, focus_id)

        record: dict[str, Any] = {
            "schema_version": 1,
            "focus_id": focus_id,
            "label": payload.get("label", ""),
            "revision": 1,
            "status": "open",
            "scope": {
                "expedition": scope.expedition,
                "leg": scope.leg,
            },
            "source": source,
            "question": payload.get("question", ""),
            "observation": payload.get("observation", ""),
            "hypothesis_text": payload.get("hypothesis_text", ""),
            "test_contract": copy.deepcopy(payload.get("test_contract")),
            "created_at": now,
            "updated_at": now,
        }

        with record_locks(self.state_dir / "locks" / "records", [focus_id]):
            if path.exists():
                raise FocusIntegrityError(path, "Focus record already exists")
            atomic_json_write(path, record)
        return record

    def update(
        self,
        scope: Scope,
        focus_id: str,
        expected_revision: int,
        changes: dict,
    ) -> dict:
        """Apply an allowed edit if the stored revision is current."""
        focus_id = self._validate_focus_id(focus_id)
        self._validate_expected_revision(expected_revision)
        self._validate_changes(changes)
        path = self._record_path(scope, focus_id)

        with record_locks(self.state_dir / "locks" / "records", [focus_id]):
            current = self._read_record(path, scope, focus_id)
            if current["status"] == "archived":
                raise FocusValidationError("archived Focus records cannot be updated")
            if current["revision"] != expected_revision:
                raise FocusConflict(current)

            for key, value in changes.items():
                current[key] = copy.deepcopy(value)
            current["revision"] += 1
            current["updated_at"] = self._whole_second_utc()
            atomic_json_write(path, current)
        return current

    def archive(
        self, scope: Scope, focus_id: str, expected_revision: int
    ) -> dict:
        """Archive a Focus without deleting its record."""
        focus_id = self._validate_focus_id(focus_id)
        self._validate_expected_revision(expected_revision)
        path = self._record_path(scope, focus_id)

        with record_locks(self.state_dir / "locks" / "records", [focus_id]):
            current = self._read_record(path, scope, focus_id)
            if current["status"] == "archived":
                raise FocusValidationError("Focus is already archived")
            if current["revision"] != expected_revision:
                raise FocusConflict(current)

            current["status"] = "archived"
            current["revision"] += 1
            current["updated_at"] = self._whole_second_utc()
            atomic_json_write(path, current)
        return current

    def _scope_dir(self, scope: Scope) -> Path:
        try:
            expedition = self._scope_component(scope.expedition, "expedition")
            leg = self._scope_component(scope.leg, "leg")
        except AttributeError as exc:
            raise FocusValidationError("scope must contain expedition and leg") from exc
        return self.state_dir / "foci" / expedition / leg

    @staticmethod
    def _scope_component(value: str, kind: str) -> str:
        from clawmarks.durable_records import validate_component

        try:
            return validate_component(value, kind)
        except (TypeError, ValueError) as exc:
            raise FocusValidationError(str(exc)) from exc

    def _record_path(self, scope: Scope, focus_id: str) -> Path:
        focus_id = self._validate_focus_id(focus_id)
        return self._scope_dir(scope) / f"{focus_id}.json"

    @staticmethod
    def _validate_focus_id(focus_id: str) -> str:
        if not isinstance(focus_id, str) or _FOCUS_ID_RE.fullmatch(focus_id) is None:
            raise FocusValidationError(
                "focus_id must match focus_<32 lowercase hexadecimal characters>"
            )
        return focus_id

    @staticmethod
    def _validate_expected_revision(expected_revision: int) -> None:
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
            raise FocusValidationError("expected_revision must be an integer")

    @staticmethod
    def _validate_changes(changes: dict) -> None:
        if not isinstance(changes, dict):
            raise FocusValidationError("changes must be an object")
        unsupported = set(changes) - _ALLOWED_UPDATE_KEYS
        if unsupported:
            key = sorted((str(item) for item in unsupported))[0]
            raise FocusValidationError(f"unsupported update key: {key}")

    def _validate_map_source(
        self, scope: Scope, source: dict[str, Any], manifest: List[dict]
    ) -> dict[str, Any]:
        if source.get("view") != "map" or source.get("kind") != "map_members":
            raise FocusValidationError("Focus source must be map_members")

        member_tags = self._deduplicate_tags(source.get("member_tags"), "member")
        if not member_tags:
            raise FocusValidationError("map Focus requires at least one member tag")
        self._validate_manifest_members(scope, member_tags, manifest)

        raw_anchor_tags = source.get("real_anchor_tags", [])
        if raw_anchor_tags is None:
            raw_anchor_tags = []
        real_anchor_tags = self._deduplicate_tags(raw_anchor_tags, "real anchor")
        self._validate_real_anchors(real_anchor_tags)

        normalized = copy.deepcopy(source)
        normalized["member_tags"] = member_tags
        normalized["real_anchor_tags"] = real_anchor_tags
        return normalized

    def _validate_frontier_source(
        self,
        scope: Scope,
        source: dict[str, Any],
        manifest: List[dict],
        coverage_cells: List[dict],
    ) -> dict[str, Any]:
        if (
            source.get("view") != "coverage"
            or source.get("kind") != "coverage_frontier"
        ):
            raise FocusValidationError("Focus source must be coverage_frontier")

        score_ranges = source.get("score_ranges")
        if not isinstance(score_ranges, dict):
            raise FocusValidationError("score_ranges must be an object")
        faith_range = self._validate_metric_range(
            score_ranges.get("faithfulness"), "faithfulness", -1.0, 1.0
        )
        novelty_range = self._validate_metric_range(
            score_ranges.get("novelty"), "novelty", 0.0, 2.0
        )

        adjacent_member_tags = self._deduplicate_tags(
            source.get("adjacent_member_tags"), "adjacent member"
        )
        if not adjacent_member_tags:
            raise FocusValidationError(
                "frontier Focus requires at least one adjacent member tag"
            )
        self._validate_manifest_members(scope, adjacent_member_tags, manifest)

        raw_anchor_tags = source.get("real_anchor_tags", [])
        if raw_anchor_tags is None:
            raw_anchor_tags = []
        real_anchor_tags = self._deduplicate_tags(raw_anchor_tags, "real anchor")
        self._validate_real_anchors(real_anchor_tags)

        self._validate_coverage_cell(coverage_cells, faith_range, novelty_range)

        normalized = copy.deepcopy(source)
        normalized["score_ranges"] = {
            "faithfulness": faith_range,
            "novelty": novelty_range,
        }
        normalized["adjacent_member_tags"] = adjacent_member_tags
        normalized["real_anchor_tags"] = real_anchor_tags
        return normalized

    @staticmethod
    def _validate_metric_range(
        value: Any, metric: str, low: float, high: float
    ) -> List[float]:
        if not isinstance(value, list) or len(value) != 2:
            raise FocusValidationError(
                f"score_ranges.{metric} must contain exactly two values"
            )
        lo, hi = value
        if (
            isinstance(lo, bool)
            or isinstance(hi, bool)
            or not isinstance(lo, (int, float))
            or not isinstance(hi, (int, float))
        ):
            raise FocusValidationError(
                f"score_ranges.{metric} must contain finite numbers"
            )
        lo_f = float(lo)
        hi_f = float(hi)
        if (
            lo_f != lo_f
            or hi_f != hi_f
            or lo_f in (float("inf"), float("-inf"))
            or hi_f in (float("inf"), float("-inf"))
        ):
            raise FocusValidationError(
                f"score_ranges.{metric} must contain finite numbers"
            )
        if not (lo_f < hi_f):
            raise FocusValidationError(
                f"score_ranges.{metric} must satisfy min < max"
            )
        if not (low <= lo_f and hi_f <= high):
            raise FocusValidationError(
                f"score_ranges.{metric} must lie within [{low}, {high}]"
            )
        return [lo_f, hi_f]

    @staticmethod
    def _validate_coverage_cell(
        coverage_cells: List[dict],
        faith_range: List[float],
        novelty_range: List[float],
    ) -> None:
        for cell in coverage_cells:
            if not isinstance(cell, dict):
                continue
            if (
                cell.get("faith_lo") == faith_range[0]
                and cell.get("faith_hi") == faith_range[1]
                and cell.get("novelty_lo") == novelty_range[0]
                and cell.get("novelty_hi") == novelty_range[1]
            ):
                if cell.get("count") != 0:
                    raise FocusValidationError(
                        "frontier Focus requires an empty coverage cell"
                    )
                if cell.get("frontier") is not True:
                    raise FocusValidationError(
                        "frontier Focus requires frontier=true on its coverage cell"
                    )
                return
        raise FocusValidationError(
            "frontier Focus requires a matching empty frontier coverage cell"
        )

    @staticmethod
    def _deduplicate_tags(value: Any, kind: str) -> List[str]:
        if not isinstance(value, list):
            raise FocusValidationError(f"{kind} tags must be a list")
        result: List[str] = []
        seen: set[str] = set()
        for tag in value:
            if not isinstance(tag, str) or not tag:
                raise FocusValidationError(f"{kind} tags must be non-empty strings")
            if tag not in seen:
                seen.add(tag)
                result.append(tag)
        return result

    def _validate_manifest_members(
        self, scope: Scope, member_tags: List[str], manifest: List[dict]
    ) -> None:
        if not isinstance(manifest, list):
            raise FocusValidationError("manifest must be a list")

        by_tag: dict[str, List[dict]] = {}
        for record in manifest:
            if not isinstance(record, dict) or not isinstance(record.get("tag"), str):
                raise FocusValidationError("manifest records must contain string tags")
            by_tag.setdefault(record["tag"], []).append(record)

        leg_root = Path(config.leg_dir(scope.expedition, scope.leg)).resolve()
        for tag in member_tags:
            matches = by_tag.get(tag, [])
            if len(matches) != 1:
                raise FocusValidationError(
                    f"generated tag {tag!r} must resolve exactly once in scoped manifest"
                )
            file_value = matches[0].get("file")
            if not isinstance(file_value, str) or not file_value:
                raise FocusIntegrityError(
                    leg_root / str(file_value),
                    f"manifest file for tag {tag!r} is invalid",
                )

            manifest_path = Path(file_value)
            candidate = (
                manifest_path
                if manifest_path.is_absolute()
                else leg_root / manifest_path
            )
            resolved = candidate.resolve()
            if not self._is_child_of(resolved, leg_root):
                raise FocusIntegrityError(
                    resolved,
                    f"manifest file for tag {tag!r} is outside the scoped leg",
                )
            if not resolved.is_file():
                raise FocusIntegrityError(
                    resolved,
                    f"manifest file for tag {tag!r} does not exist",
                )

    def _validate_real_anchors(self, tags: List[str]) -> None:
        root = self.real_dir.resolve()
        for tag in tags:
            candidate_name = Path(tag)
            if (
                candidate_name.is_absolute()
                or len(candidate_name.parts) != 1
                or candidate_name.name != tag
                or tag in (".", "..")
            ):
                raise FocusValidationError(f"invalid real anchor tag: {tag!r}")
            resolved = (self.real_dir / candidate_name).resolve()
            if not self._is_child_of(resolved, root) or not resolved.is_file():
                raise FocusValidationError(f"unknown real anchor: {tag}")

    @staticmethod
    def _is_child_of(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return path != root

    @staticmethod
    def _whole_second_utc() -> str:
        return datetime.datetime.fromisoformat(utc_now()).replace(
            microsecond=0
        ).isoformat()

    def _read_record(
        self, path: Path, scope: Scope, focus_id: str
    ) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
        except FileNotFoundError as exc:
            raise FocusNotFound(focus_id) from exc
        except json.JSONDecodeError as exc:
            raise FocusIntegrityError(path, f"invalid JSON: {exc}") from exc
        except (OSError, UnicodeError) as exc:
            raise FocusIntegrityError(path, f"could not read record: {exc}") from exc

        if not isinstance(record, dict):
            raise FocusIntegrityError(path, "record JSON must contain an object")
        self._validate_stored_record(path, scope, focus_id, record)
        return record

    @staticmethod
    def _validate_stored_record(
        path: Path, scope: Scope, focus_id: str, record: dict[str, Any]
    ) -> None:
        if record.get("schema_version") != 1:
            raise FocusIntegrityError(path, "unsupported Focus schema version")
        if record.get("focus_id") != focus_id:
            raise FocusIntegrityError(path, "record focus_id does not match its filename")
        if record.get("scope") != {
            "expedition": scope.expedition,
            "leg": scope.leg,
        }:
            raise FocusIntegrityError(path, "record scope does not match its directory")
        revision = record.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise FocusIntegrityError(path, "record revision is invalid")
        if record.get("status") not in ("open", "archived"):
            raise FocusIntegrityError(path, "record status is invalid")
