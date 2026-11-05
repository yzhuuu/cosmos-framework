"""Validation labels must never be exported into training, or vice versa."""

import pytest

from examples.cosmos_lab.export_reviewed_preferences import annotation_eligible


def annotation(**changes):
    return dict(
        decision="a",
        confidence="medium",
        all_frames_reviewed=True,
        evidence=[{"frames": [0, 10]}],
        reviewer_type="agent",
        **changes,
    )


def test_validation_requires_explicit_purpose_and_excludes_train():
    row = annotation(training_eligible=False, validation_eligible=True)
    assert annotation_eligible(row, "validation")
    with pytest.raises(ValueError, match="Training cannot"):
        annotation_eligible(row, "train")


def test_train_cannot_be_validation():
    row = annotation(training_eligible=True)
    assert annotation_eligible(row, "train")
    with pytest.raises(ValueError, match="Validation cannot"):
        annotation_eligible(row, "validation")


@pytest.mark.parametrize("purpose", ["train", "validation"])
def test_abstention_not_exported(purpose):
    row = annotation(training_eligible=False, validation_eligible=False)
    row["decision"] = "abstain"
    assert not annotation_eligible(row, purpose)


@pytest.mark.parametrize("field,value", [("all_frames_reviewed", False), ("evidence", []), ("confidence", "low")])
def test_incomplete_validation_review_rejected(field, value):
    row = annotation(training_eligible=False, validation_eligible=True)
    row[field] = value
    with pytest.raises(ValueError, match="Insufficient"):
        annotation_eligible(row, "validation")
