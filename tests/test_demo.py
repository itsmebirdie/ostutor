import pytest

from ostutor.slice import demo


@pytest.mark.parametrize("argv", [["--all"], ["deadlock_two_locks", "--tamper"],
                                  ["--scaling"], ["--rr"], ["--scaling", "--rr"]])
def test_demo_modes_run_cleanly(argv, capsys):
    assert demo.main(argv) in (0, None)
    assert capsys.readouterr().out.strip()
