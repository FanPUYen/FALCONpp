"""Run real KeyGen -> Sign -> Verify for the manuscript's selected profiles."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "implement"))
from falconpp.scheme import FalconPlusPlus
from falconpp.parameters import coefficient_entropy_bits
from falconpp.rans import encode_raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter", choices=["I-1245", "V-117", "both"], default="both")
    parser.add_argument("--message", default="FALCON++ reproducibility example")
    parser.add_argument("--wire-format", choices=["raw", "padded"], default="raw",
                        help="raw: unpadded canonical rANS (default); padded: conservative fixed-length diagnostic.")
    parser.add_argument("--seed", help="Optional public experiment seed for repeatable runs.")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "workflow.json")
    args = parser.parse_args()
    names = ["I-1245", "V-117"] if args.parameter == "both" else [args.parameter]
    message = args.message.encode("utf-8")
    rows = []
    for name in names:
        print(f"{name}: KeyGen", flush=True)
        padded = args.wire_format == "padded"
        scheme = FalconPlusPlus(name, formal_signatures=padded, allow_provisional_wire=padded)
        def seed(stage):
            return None if args.seed is None else json.dumps(
                [args.seed, name, stage], ensure_ascii=True).encode("ascii")
        start = perf_counter()
        keys = scheme.keygen(seed=seed("keygen"))
        keygen_s = perf_counter() - start
        print(f"{name}: Sign (includes lazy sampler/correction setup)", flush=True)
        start = perf_counter()
        signed = scheme.sign_detailed(keys.secret_key, message, seed=seed("sign"))
        sign_s = perf_counter() - start
        wire = signed.wire
        start = perf_counter()
        valid = scheme.verify(bytes(keys.public_key), message, wire)
        verify_s = perf_counter() - start
        changed_message_rejected = not scheme.verify(
            bytes(keys.public_key), message + b" [modified]", wire)
        if not valid or not changed_message_rejected:
            raise RuntimeError(f"End-to-end verification failed for {name}")
        raw_payload = encode_raw(signed.s2, scheme.codec.model, norm_bound=scheme.parameters.beta)
        salt_bytes = scheme.parameters.salt_bytes
        row = {
            "parameter": name, "n": scheme.parameters.n, "q": scheme.parameters.q,
            "valid_signature": valid, "modified_message_rejected": changed_message_rejected,
            "public_key_payload_bytes": len(bytes(keys.public_key)),
            "signature_wire_bytes": len(wire),
            "paper_entropy_estimate_bytes": salt_bytes + math.ceil(
                scheme.parameters.n * coefficient_entropy_bits(scheme.parameters.sigma_sig) / 8),
            "raw_rans_wire_bytes": salt_bytes + len(raw_payload),
            "conservative_padded_wire_bytes": salt_bytes + scheme.codec.length_analysis.certified_upper_bound,
            "added_padding_bytes_in_selected_format": len(wire) - salt_bytes - len(raw_payload),
            "wire_format": args.wire_format,
            "wire_policy": scheme.codec.fixed_length_policy,
            "keygen_sampled_pairs": keys.statistics.sampled_pairs,
            "signing_trials": signed.statistics.total_trials,
            "correction_rejections": signed.statistics.correction_rejections,
            "norm_rejections": signed.statistics.norm_rejections,
            "seconds": {"keygen": keygen_s, "first_sign": sign_s, "verify": verify_s},
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "scope": "Research implementation. Default raw mode uses canonical rANS without fixed padding. The optional padded format uses a conservative bound. Paper entropy estimates, actual compressed length, and padded length are reported separately. One sample is not a benchmark.",
        "randomness": "OS-seeded" if args.seed is None else "public deterministic experiment seed",
        "experiment_seed": args.seed,
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Workflow passed; summary: {args.output}")


if __name__ == "__main__":
    main()
