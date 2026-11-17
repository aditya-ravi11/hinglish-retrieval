import json

from hinglish_retrieval.eval import models


def _make(tmp_path, key):
    d = tmp_path / key
    d.mkdir()
    spec = {"licence": "MIT", "query_prompt": "query: ", "doc_prompt": "passage: ", "max_len": 512}
    (d / "hr_spec.json").write_text(json.dumps(spec))


def test_dev_sees_all_test_sees_only_selected(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(models, "SELECTED", tmp_path / "SELECTED")
    _make(tmp_path, "hr-a")
    _make(tmp_path, "hr-b")
    assert sorted(models.trained("dev")) == ["hr-a", "hr-b"]
    assert models.trained("test") == {}  # nothing selected yet
    (tmp_path / "SELECTED").write_text("hr-b\n")
    t = models.trained("test")
    assert list(t) == ["hr-b"] and t["hr-b"].hf_id == str(tmp_path / "hr-b")
    assert t["hr-b"].query_prompt == "query: "
