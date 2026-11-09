"""Fixture-only checks: no actual images, validation or official-test access."""
from pathlib import Path
import json

import replay_oracle_probe as probe


if __name__ == "__main__":
    result = probe.fixture()
    result["source_sha256"] = probe.sha(Path(probe.__file__))
    result["check_source_sha256"] = probe.sha(__file__)
    probe.runner.atomic_json(Path(__file__).parent / "oracle_replay_fixture_check.json", result)
    print(json.dumps(result, indent=2))
