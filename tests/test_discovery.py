import pytest

from fmriprepviz.discovery import NoBoldRefsError, find_boldrefs, parse_entities


def test_parses_bids_entities_from_a_filename():
    entities = parse_entities(
        "sub-01_ses-02_task-rest_run-10_space-T1w_boldref.nii.gz"
    )
    assert entities["sub"] == "01"
    assert entities["ses"] == "02"
    assert entities["task"] == "rest"
    assert entities["run"] == "10"
    assert entities["space"] == "T1w"


def test_finds_every_t1w_boldref(derivatives):
    refs = find_boldrefs(derivatives, "01")
    assert len(refs) == 5


def test_ignores_boldrefs_in_other_spaces(derivatives):
    refs = find_boldrefs(derivatives, "01")
    assert all(ref.entities["space"] == "T1w" for ref in refs)


def test_orders_by_session_then_task_then_run_numerically(derivatives):
    refs = find_boldrefs(derivatives, "01")
    ordered = [(r.entities["ses"], r.entities["task"], r.entities["run"]) for r in refs]
    assert ordered == [
        ("01", "rest", "1"),
        ("01", "rest", "2"),
        ("01", "rest", "10"),
        ("02", "nback", "1"),
        ("02", "rest", "1"),
    ]


def test_label_reads_as_human_readable_entities(derivatives):
    refs = find_boldrefs(derivatives, "01")
    assert refs[0].label == "ses-01 task-rest run-1"


def test_filters_by_session(derivatives):
    refs = find_boldrefs(derivatives, "01", ses=["02"])
    assert len(refs) == 2
    assert all(ref.entities["ses"] == "02" for ref in refs)


def test_filters_by_task(derivatives):
    refs = find_boldrefs(derivatives, "01", task=["nback"])
    assert len(refs) == 1


def test_filters_by_run(derivatives):
    refs = find_boldrefs(derivatives, "01", run=["1"])
    assert len(refs) == 3


def test_filters_combine(derivatives):
    refs = find_boldrefs(derivatives, "01", ses=["01"], run=["2"])
    assert len(refs) == 1
    assert refs[0].entities["run"] == "2"


def test_missing_t1w_output_names_the_spaces_that_are_present(
    derivatives_without_t1w,
):
    with pytest.raises(NoBoldRefsError) as excinfo:
        find_boldrefs(derivatives_without_t1w, "01")
    message = str(excinfo.value)
    assert "MNI152NLin2009cAsym" in message
    assert "T1w" in message


def test_unknown_subject_reports_the_directory_searched(derivatives):
    with pytest.raises(NoBoldRefsError) as excinfo:
        find_boldrefs(derivatives, "99")
    assert "sub-99" in str(excinfo.value)


def test_over_restrictive_filters_report_the_filters(derivatives):
    with pytest.raises(NoBoldRefsError) as excinfo:
        find_boldrefs(derivatives, "01", task=["nosuchtask"])
    assert "nosuchtask" in str(excinfo.value)
