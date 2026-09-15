"""Pastikan endpoint serving menghasilkan prediksi yang sama dengan model training."""

import argparse
import json
from pathlib import Path
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5001")
    parser.add_argument("--output-dir", type=Path, default=Path("MLProject/outputs"))
    args = parser.parse_args()
    for attempt in range(60):
        try:
            with urlopen(f"{args.url}/ping", timeout=3) as response:
                if response.status == 200:
                    break
        except (URLError, TimeoutError):
            pass
        time.sleep(2)
    else:
        raise RuntimeError("Endpoint serving belum siap setelah 60 pemeriksaan.")
    request = Request(f"{args.url}/invocations", data=(args.output_dir / "request.json").read_bytes(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=30) as response:
        result = json.load(response)
    expected = json.loads((args.output_dir / "expected_predictions.json").read_text())
    if result.get("predictions") != expected:
        raise AssertionError(f"Prediksi serving berbeda: {result}")
    (args.output_dir / "serving_verification.json").write_text(
        json.dumps({"url": args.url, "samples": len(expected), "matched": True, "response": result}, indent=2), encoding="utf-8")
    print(f"Serving berhasil: {len(expected)} prediksi cocok dengan model training.")


if __name__ == "__main__":
    main()
