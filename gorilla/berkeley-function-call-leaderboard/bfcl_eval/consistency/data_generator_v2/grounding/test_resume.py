"""Offline checks for preserving accepted work and retrying failed cases."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import ground_and_replay as pipeline


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        self.source.mkdir()
        self.output = self.root / "output"
        self.backend = self.root / "backend.py"
        self.backend.write_text("# stub backend", encoding="utf-8")
        self.tools = self.root / "tools.json"
        self.tools.write_text(json.dumps([{"name": "read_item", "parameters": {}}]), encoding="utf-8")
        self.args = ["--input-dir", str(self.source), "--tool-doc", str(self.tools),
                     "--backend", str(self.backend), "--output-dir", str(self.output),
                     "--no-memory", "--max-tokens", "32768", "--request-timeout", "300"]

    def add_source(self, name):
        (self.source / f"{name}.json").write_text(json.dumps({"id": name}), encoding="utf-8")

    def run_cli(self, responses, *, resume=False):
        with patch.object(pipeline, "ground_one", side_effect=responses) as ground:
            with contextlib.redirect_stdout(io.StringIO()):
                code = pipeline.main(self.args + (["--resume"] if resume else []))
        return code, ground

    def read_manifest(self):
        return json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))

    def test_resume_preserves_success_and_retries_all_failure_types_before_pending(self):
        for name in ("a", "c", "d"):
            self.add_source(name)
        code, _ = self.run_cli([{"status": "accepted"}, TimeoutError("timed out"),
                                {"status": "rejected", "errors": ["branch mismatch"]}])
        self.assertEqual(code, 1)
        accepted = (self.output / "a.json").read_bytes()
        old_manifest = self.read_manifest()
        self.add_source("b")  # An unfinished case that sorts before the failures.
        code, calls = self.run_cli([{"status": "accepted"}] * 3, resume=True)
        self.assertEqual(code, 0)
        self.assertEqual([call.args[0]["id"] for call in calls.call_args_list], ["c", "d", "b"])
        self.assertTrue(all(call.kwargs["request_timeout"] == 300 for call in calls.call_args_list))
        self.assertTrue(all(call.kwargs["max_tokens"] == 32768 for call in calls.call_args_list))
        self.assertEqual((self.output / "a.json").read_bytes(), accepted)
        manifest = self.read_manifest()
        self.assertEqual((manifest["requested"], manifest["accepted"], manifest["rejected"]), (4, 4, 0))
        self.assertEqual(manifest["failures"], [])
        self.assertEqual(manifest["resume"]["retry_failed_sources"], ["c.json", "d.json"])
        self.assertEqual(manifest["resume"]["retained_accepted"], 1)
        self.assertEqual(manifest["resume"]["scheduled"], 3)
        self.assertNotIn("current_source", manifest)
        snapshot = self.output / manifest["resume"]["previous_manifest"]
        self.assertEqual(json.loads(snapshot.read_text(encoding="utf-8")), old_manifest)
        self.assertEqual(json.loads((snapshot.parent / "d.json").read_text())["status"], "rejected")

    def test_resume_recovers_saved_success_before_manifest_update(self):
        self.add_source("a")
        self.run_cli([{"status": "accepted"}])
        manifest = self.read_manifest()
        manifest.update(status="running", accepted=0, cases=[])
        pipeline._write_json(self.output / "manifest.json", manifest)
        code, calls = self.run_cli([], resume=True)
        self.assertEqual(code, 0)
        calls.assert_not_called()
        self.assertEqual(self.read_manifest()["accepted"], 1)

    def test_failed_retry_replaces_old_rejected_artifact_and_counts_once(self):
        self.add_source("a")
        self.run_cli([{"status": "rejected"}])
        code, calls = self.run_cli([TimeoutError("timed out again")], resume=True)
        self.assertEqual(code, 1)
        self.assertEqual(calls.call_count, 1)
        manifest = self.read_manifest()
        self.assertEqual(manifest["rejected"], 1)
        self.assertEqual(len(manifest["failures"]), 1)
        saved = json.loads((self.output / "a.json").read_text())
        self.assertEqual(saved["status"], "failed")
        self.assertIn("timed out again", saved["error"])

    def test_nonempty_directory_requires_resume_and_different_input_is_refused(self):
        self.add_source("a")
        self.run_cli([{"status": "accepted"}])
        original = (self.output / "manifest.json").read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline.main(self.args)
        other = self.root / "other"
        other.mkdir()
        self.args[self.args.index("--input-dir") + 1] = str(other)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline.main(self.args + ["--resume"])
        self.assertEqual((self.output / "manifest.json").read_bytes(), original)

    def test_timeout_is_forwarded_to_client(self):
        response = {"choices": [{"message": {"content": "{}"}}]}
        with patch.object(pipeline, "chat", return_value=response) as chat:
            self.assertEqual(pipeline._model_json([], model="test", max_tokens=32768, timeout=300), {})
        self.assertEqual(chat.call_args.kwargs["timeout"], 300)
        self.assertEqual(chat.call_args.kwargs["max_tokens"], 32768)

    def test_atomic_write_retries_temporary_file_access_errors(self):
        path = self.root / "manifest.json"
        pipeline._write_json(path, {"accepted": 37})
        replace = pipeline.os.replace
        attempts = 0

        def temporarily_locked(source, destination):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                self.assertEqual(json.loads(path.read_text()), {"accepted": 37})
                raise PermissionError("target temporarily in use")
            replace(source, destination)

        with patch.object(pipeline.os, "replace", side_effect=temporarily_locked), patch.object(pipeline, "sleep") as sleep:
            pipeline._write_json(path, {"accepted": 38})
        self.assertEqual(attempts, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.1, 0.2])
        self.assertEqual(json.loads(path.read_text()), {"accepted": 38})
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_persistent_file_access_error_is_bounded_and_preserves_previous_file(self):
        path = self.root / "manifest.json"
        pipeline._write_json(path, {"accepted": 37})
        with patch.object(pipeline.os, "replace", side_effect=PermissionError("denied")) as replace:
            with patch.object(pipeline, "sleep") as sleep, self.assertRaises(PermissionError):
                pipeline._write_json(path, {"accepted": 38})
        self.assertEqual(replace.call_count, 6)
        self.assertEqual(sleep.call_count, 5)
        self.assertEqual(json.loads(path.read_text()), {"accepted": 37})
        self.assertEqual(list(self.root.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
