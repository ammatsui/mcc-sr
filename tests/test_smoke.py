import subprocess
import sys

def test_cli_runs():
    result = subprocess.run(
        [sys.executable, "-m", "mc_symbolic_regression.cli", "mwe"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "MWE:" in result.stdout

def test_mse_sanity():
    result = subprocess.run(
        [sys.executable, "-m", "mc_symbolic_regression.cli", "mwe"],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if "MSE" in line:
            mse = float(line.rsplit("=", 1)[-1])
            assert 0 < mse < 0.05
            break
    else:
        assert False, "MSE output not found"