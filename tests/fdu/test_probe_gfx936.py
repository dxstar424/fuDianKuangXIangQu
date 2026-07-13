import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/probe_gfx936.py"


def _load_module():
    name = "probe_gfx936_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load probe module")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class ProbeGfx936Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_probe_is_sequential_deterministic_and_ordered(self) -> None:
        requests = []

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            index = len(requests)
            return _Response(
                {
                    "choices": [
                        {
                            "message": {"content": f"answer-{index}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens": index},
                }
            )

        with mock.patch.object(self.module.urllib.request, "urlopen", fake_urlopen):
            result = self.module.run_probe(
                host="127.0.0.1", port=8123, model="model", label="candidate"
            )

        self.assertEqual(result["prompts"], list(self.module.PROMPTS))
        self.assertEqual(result["responses"], ["answer-1", "answer-2", "answer-3"])
        self.assertEqual(result["finish_reasons"], ["stop", "stop", "stop"])
        self.assertEqual(len(requests), 3)
        for index, (request, timeout) in enumerate(requests):
            payload = json.loads(request.data)
            self.assertEqual(request.full_url, "http://127.0.0.1:8123/v1/chat/completions")
            self.assertEqual(timeout, 300)
            self.assertEqual(payload["messages"][0]["content"], self.module.PROMPTS[index])
            self.assertEqual(payload["temperature"], 0.0)
            self.assertEqual(payload["seed"], 20260714)
            self.assertEqual(payload["max_tokens"], 64)
            self.assertFalse(payload["stream"])


if __name__ == "__main__":
    unittest.main()
