"""Tests for src/core/tool_factory.py (Round 8, item 8.2)."""
from __future__ import annotations

from pathlib import Path

from src.core.tool_factory import (
    GeneratedToolSpec,
    compute_dataset_fingerprint,
    load_persisted_tools,
    make_spec,
    next_version,
    register_and_persist,
    validate_spec,
)
from tests.fixtures import single_column

VALID_SCHEMA = {
    "threshold": {"type": "number", "description": "a threshold", "required": False}
}
VALID_CODE = "RESULT = threshold if 'threshold' in dir() else 1\n"


class TestValidateSpecName:
    def test_valid_snake_case_passes(self) -> None:
        errors = validate_spec("row_counter", "counts rows", {}, "RESULT = 1\n", [], {})
        assert errors == []

    def test_camel_case_rejected(self) -> None:
        errors = validate_spec("RowCounter", "counts rows", {}, "RESULT = 1\n", [], {})
        assert any("snake_case" in e for e in errors)

    def test_hyphenated_rejected(self) -> None:
        errors = validate_spec("row-counter", "x", {}, "RESULT = 1\n", [], {})
        assert any("snake_case" in e for e in errors)

    def test_leading_digit_rejected(self) -> None:
        errors = validate_spec("2cool", "x", {}, "RESULT = 1\n", [], {})
        assert any("snake_case" in e for e in errors)

    def test_double_underscore_rejected(self) -> None:
        errors = validate_spec("row__counter", "x", {}, "RESULT = 1\n", [], {})
        assert any("double underscore" in e for e in errors)

    def test_trailing_underscore_rejected(self) -> None:
        errors = validate_spec("row_counter_", "x", {}, "RESULT = 1\n", [], {})
        assert any("end with an underscore" in e for e in errors)

    def test_builtin_name_collision_rejected(self) -> None:
        errors = validate_spec(
            "clean_data", "malicious", {}, "RESULT = 1\n", ["clean_data", "train_model"], {}
        )
        assert any("built-in tool name" in e for e in errors)

    def test_generated_name_reuse_is_not_a_collision(self) -> None:
        # A name already held by a previous generation is a modification,
        # not a collision — validate_spec only rejects BUILT-IN collisions
        # (existing_tool_names), the caller is expected to exclude already-
        # generated names from that list when this is a resubmission.
        errors = validate_spec("row_counter", "v2", {}, "RESULT = 2\n", [], {})
        assert errors == []


class TestValidateSpecDescriptionAndSchema:
    def test_empty_description_rejected(self) -> None:
        errors = validate_spec("row_counter", "", {}, "RESULT = 1\n", [], {})
        assert any("description" in e for e in errors)

    def test_whitespace_only_description_rejected(self) -> None:
        errors = validate_spec("row_counter", "   ", {}, "RESULT = 1\n", [], {})
        assert any("description" in e for e in errors)

    def test_non_dict_schema_rejected(self) -> None:
        errors = validate_spec("row_counter", "x", ["not", "a", "dict"], "RESULT = 1\n", [], {})  # type: ignore[arg-type]
        assert any("params_schema must be a dict" in e for e in errors)

    def test_schema_entry_missing_keys_rejected(self) -> None:
        errors = validate_spec(
            "row_counter", "x", {"threshold": {"type": "number"}}, "RESULT = 1\n", [], {}
        )
        assert any("missing keys" in e for e in errors)

    def test_schema_entry_non_bool_required_rejected(self) -> None:
        errors = validate_spec(
            "row_counter",
            "x",
            {"threshold": {"type": "number", "description": "d", "required": "yes"}},
            "RESULT = 1\n",
            [],
            {},
        )
        assert any("must be a bool" in e for e in errors)

    def test_schema_param_shadowing_reserved_name_rejected(self) -> None:
        for reserved in ("df", "SCHEMA", "RESULT", "FINDING"):
            errors = validate_spec(
                "row_counter",
                "x",
                {reserved: {"type": "number", "description": "d", "required": False}},
                "RESULT = 1\n",
                [],
                {},
            )
            assert any("reserved sandbox" in e for e in errors), f"{reserved} should be rejected"

    def test_valid_schema_passes(self) -> None:
        errors = validate_spec("row_counter", "x", VALID_SCHEMA, VALID_CODE, [], {})
        assert errors == []


class TestValidateSpecCode:
    def test_empty_code_rejected(self) -> None:
        errors = validate_spec("row_counter", "x", {}, "", [], {})
        assert any("code must be" in e for e in errors)

    def test_code_missing_result_rejected(self) -> None:
        errors = validate_spec("row_counter", "x", {}, "x = 1\n", [], {})
        assert any("static check" in e for e in errors)

    def test_code_disallowed_import_rejected(self) -> None:
        errors = validate_spec("row_counter", "x", {}, "import os\nRESULT = 1\n", [], {})
        assert any("static check" in e for e in errors)

    def test_all_errors_accumulate(self) -> None:
        # Bad name + empty description + bad schema + empty code all report
        # together, not just the first one found.
        errors = validate_spec("Bad Name", "", ["nope"], "", [], {})  # type: ignore[arg-type]
        assert len(errors) >= 4


class TestComputeDatasetFingerprint:
    def test_same_file_same_fingerprint(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        assert compute_dataset_fingerprint(str(dataset)) == compute_dataset_fingerprint(str(dataset))

    def test_different_files_different_fingerprints(self, tmp_path: Path) -> None:
        a_dir = tmp_path / "a"
        a_dir.mkdir()
        b_dir = tmp_path / "b"
        b_dir.mkdir()
        a = single_column(a_dir)
        b = single_column(b_dir)
        assert compute_dataset_fingerprint(str(a)) != compute_dataset_fingerprint(str(b))

    def test_missing_file_falls_back_gracefully(self, tmp_path: Path) -> None:
        fp = compute_dataset_fingerprint(str(tmp_path / "does_not_exist.csv"))
        assert isinstance(fp, str)
        assert fp  # non-empty, never raises


class TestNextVersion:
    def test_new_name_is_version_1(self) -> None:
        assert next_version({}, "row_counter") == 1

    def test_existing_name_increments(self) -> None:
        prior = GeneratedToolSpec(
            name="row_counter", description="x", params_schema={}, code="RESULT=1\n",
            version=3, created_at="t", dataset_fingerprint="fp",
        )
        assert next_version({"row_counter": prior}, "row_counter") == 4


class TestMakeSpec:
    def test_stamps_version_and_timestamp(self) -> None:
        spec = make_spec("row_counter", "x", {}, "RESULT=1\n", "fp123")
        assert spec.name == "row_counter"
        assert spec.version == 1
        assert spec.dataset_fingerprint == "fp123"
        assert spec.created_at  # non-empty ISO string
        assert spec.source == "llm_generated"


class TestRegisterAndPersist:
    def test_registers_into_tool_registry(self, tmp_path: Path) -> None:
        class _FakeRegistry:
            def __init__(self) -> None:
                self.registered: dict[str, object] = {}

            def register(self, tool: object) -> None:
                self.registered[tool.name] = tool  # type: ignore[attr-defined]

        class _FakeMemory:
            pass  # no add_generated_tool — exercises the hasattr guard

        registry = _FakeRegistry()
        spec = make_spec("row_counter", "counts rows", {}, "RESULT = len(df)\n", "fp1")
        register_and_persist(spec, registry, _FakeMemory(), str(tmp_path))

        assert "row_counter" in registry.registered
        assert registry.registered["row_counter"].name == "row_counter"  # type: ignore[attr-defined]

    def test_calls_memory_add_generated_tool_when_present(self, tmp_path: Path) -> None:
        class _FakeRegistry:
            def register(self, tool: object) -> None: pass

        class _FakeMemory:
            def __init__(self) -> None:
                self.added: list[dict] = []

            def add_generated_tool(self, spec_dict: dict) -> None:
                self.added.append(spec_dict)

        memory = _FakeMemory()
        spec = make_spec("row_counter", "counts rows", {}, "RESULT = len(df)\n", "fp1")
        register_and_persist(spec, _FakeRegistry(), memory, str(tmp_path))

        assert len(memory.added) == 1
        assert memory.added[0]["name"] == "row_counter"

    def test_persists_spec_to_disk(self, tmp_path: Path) -> None:
        class _FakeRegistry:
            def register(self, tool: object) -> None: pass

        class _FakeMemory:
            pass

        spec = make_spec("row_counter", "counts rows", {}, "RESULT = len(df)\n", "fp1")
        register_and_persist(spec, _FakeRegistry(), _FakeMemory(), str(tmp_path))

        spec_path = tmp_path / "generated_tools" / "row_counter.json"
        assert spec_path.exists()

    def test_second_version_keeps_first_under_history(self, tmp_path: Path) -> None:
        class _FakeRegistry:
            def register(self, tool: object) -> None: pass

        class _FakeMemory:
            pass

        v1 = make_spec("row_counter", "v1", {}, "RESULT = 1\n", "fp1")
        register_and_persist(v1, _FakeRegistry(), _FakeMemory(), str(tmp_path))

        v2 = GeneratedToolSpec(
            name="row_counter", description="v2", params_schema={}, code="RESULT = 2\n",
            version=2, created_at="t2", dataset_fingerprint="fp1",
        )
        register_and_persist(v2, _FakeRegistry(), _FakeMemory(), str(tmp_path))

        import json
        payload = json.loads((tmp_path / "generated_tools" / "row_counter.json").read_text())
        assert payload["version"] == 2
        assert len(payload["history"]) == 1
        assert payload["history"][0]["version"] == 1


class TestLoadPersistedTools:
    def test_no_directory_returns_empty(self, tmp_path: Path) -> None:
        assert load_persisted_tools(str(tmp_path), "fp1") == []

    def test_matching_fingerprint_returned(self, tmp_path: Path) -> None:
        class _FakeRegistry:
            def register(self, tool: object) -> None: pass

        class _FakeMemory:
            pass

        spec = make_spec("row_counter", "x", {}, "RESULT = 1\n", "fp-match")
        register_and_persist(spec, _FakeRegistry(), _FakeMemory(), str(tmp_path))

        loaded = load_persisted_tools(str(tmp_path), "fp-match")
        assert len(loaded) == 1
        assert loaded[0].name == "row_counter"

    def test_mismatched_fingerprint_excluded(self, tmp_path: Path) -> None:
        class _FakeRegistry:
            def register(self, tool: object) -> None: pass

        class _FakeMemory:
            pass

        spec = make_spec("row_counter", "x", {}, "RESULT = 1\n", "fp-a")
        register_and_persist(spec, _FakeRegistry(), _FakeMemory(), str(tmp_path))

        assert load_persisted_tools(str(tmp_path), "fp-b") == []
