"""Run offline tests, dependency checks, and all dashboard views on local artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_model.market_reporting import file_hash


def main():
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    print(tests.stdout, end="", flush=True)
    if tests.returncode:
        print(tests.stderr, file=sys.stderr)
        raise SystemExit(tests.returncode)
    dependencies = subprocess.run([sys.executable, "-m", "pip", "check"], cwd=ROOT, capture_output=True, text=True)
    print(dependencies.stdout, end="", flush=True)
    if dependencies.returncode:
        raise SystemExit(dependencies.returncode)
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    if app.exception:
        raise RuntimeError([item.message for item in app.exception])
    checks = {"Game markets": "passed"}
    for name in ("Player props", "Season stats", "Live odds & lines", "Archived quote replay", "Model validation", "Data audit"):
        app.radio(key="view").set_value(name).run()
        if app.exception or app.error:
            raise RuntimeError(f"Dashboard {name}: {[item.message for item in app.exception]} / {[item.value for item in app.error]}")
        checks[name] = "passed"
        print(f"Dashboard {name}: passed", flush=True)
    output = ROOT / "artifacts/verification"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "tests": tests.stdout.strip().splitlines()[-1],
                "dependencies": dependencies.stdout.strip(), "dashboard_views": checks,
                "dashboard_method": "Streamlit AppTest headless interaction with actual local datasets/models; synthetic UI tests cover edited invalid quotes.",
                "versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "nflreadpy", "streamlit")},
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for directory in (ROOT / "src", ROOT / "scripts", ROOT / "tests") for p in directory.rglob("*.py")},
                "model_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in (ROOT / "artifacts").rglob("*.joblib")}}
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "report.md").write_text("# Project verification\n\n" + metadata["tests"] + "\n\n" + metadata["dependencies"] + "\n\n" +
                                     "\n".join(f"- {name}: {result}" for name, result in checks.items()) + "\n\n" + metadata["dashboard_method"] + "\n")
    print(f"Verification report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
