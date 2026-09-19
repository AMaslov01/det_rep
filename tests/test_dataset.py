import csv
import json

import pytest

from det_rep.dataset import load_prepared, prepare, qa_sources, source_split


def fixture_inputs(tmp_path, answers=750):
    source_path = tmp_path / "source_info.jsonl"
    ids = [str(x) for x in range(1, 989)] + ["12448"]
    with source_path.open("w", encoding="utf-8") as stream:
        for source_id in ids:
            stream.write(json.dumps({
                "source_id": int(source_id), "task_type": "QA",
                "source_info": {"passages": f"Fact for {source_id}.", "question": f"Question for {source_id}?"},
            }) + "\n")
    answer_path = tmp_path / "answers.csv"
    with answer_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "id", "generated_response", "prompt", "hallucination",
            "annotation_reason", "annotation_raw", "annotation_model",
        ])
        writer.writeheader()
        for source_id in ids[:answers]:
            writer.writerow({
                "id": f"llama31_8b_{source_id}", "generated_response": f"Answer {source_id}.",
                "prompt": f"Fact for {source_id}. Question for {source_id}?",
                "hallucination": int(source_id) % 2 if source_id != "12448" else 0,
                "annotation_reason": "fixture", "annotation_raw": "{}", "annotation_model": "fixture",
            })
    return source_path, answer_path


def test_partial_and_full_answers_keep_same_frozen_989_source_split(tmp_path):
    source_path, answer_path = fixture_inputs(tmp_path, 750)
    work = tmp_path / "external-work"
    first, partial = prepare(source_path, answer_path, work)
    assert (partial["sources"], partial["available"], partial["pending"]) == (989, 750, 239)
    assert partial["train_available"] + partial["test_available"] == 750
    frozen = (work / "source_split.json").read_bytes()
    fixture_inputs(tmp_path, 989)
    second, full = prepare(source_path, answer_path, work)
    assert first != second
    assert (full["available"], full["pending"]) == (989, 0)
    assert len(load_prepared(source_path, answer_path, second)) == 989
    assert (work / "source_split.json").read_bytes() == frozen
    splits = source_split(set(qa_sources(source_path)))
    assert list(splits.values()).count("train") == 791
    assert list(splits.values()).count("test") == 198
    assert "12448" in splits


def test_rejects_changed_input_and_duplicate_source(tmp_path):
    source_path, answer_path = fixture_inputs(tmp_path, 750)
    manifest, _ = prepare(source_path, answer_path, tmp_path / "work")
    with answer_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="differ"):
        load_prepared(source_path, answer_path, manifest)
    with answer_path.open("a", encoding="utf-8") as stream:
        stream.write("llama31_8b_1,duplicate,prompt,0,reason,{},fixture\n")
    with pytest.raises(ValueError, match="duplicate"):
        prepare(source_path, answer_path, tmp_path / "work")
