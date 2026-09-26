"""Export manuscript Tables 1-6 from freshly computed numerical results."""
import csv
import json
from pathlib import Path

import mpmath as mp

ROOT = Path(__file__).resolve().parent


def export_tables():
    mp.mp.dps = 100
    computed = json.loads((ROOT / "results/selected_parameters.json").read_text(encoding="utf-8"))["rows"]
    entropy = json.loads((ROOT / "results/entropy.json").read_text(encoding="utf-8"))["runs"]
    literature = json.loads((ROOT / "data/literature_baselines.json").read_text(encoding="utf-8"))["rows"]
    sizes = {r["n"]: r["rounded_entropy_based_estimate_bytes"] for r in entropy}
    out = ROOT / "results/tables"
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    def write(number, title, header, rows, note):
        stem = f"table{number}"
        with (out / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        lines = [f"# Table {number}: {title}", "", note, "",
                 "| " + " | ".join(header) + " |",
                 "| " + " | ".join(["---"] * len(header)) + " |"]
        lines += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
        (out / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest.append({"table": number, "title": title, "csv": f"{stem}.csv", "markdown": f"{stem}.md", "scope": note})

    def fixed(x, digits, direction=None):
        x = mp.mpf(x)
        scale = mp.mpf(10) ** digits
        if direction == "lower":
            x = mp.floor(x * scale) / scale
        elif direction == "upper":
            x = mp.ceil(x * scale) / scale
        return f"{float(x):.{digits}f}"

    def pair(label, fn):
        return [label] + [fn(r) for r in computed]

    header = ["Quantity", "FALCON++-512", "FALCON++-1024"]
    write(1, "Expected sampling iterations at matched narrow-width parameters", header, [
        pair("q", lambda r: r["q"]),
        pair("gamma", lambda r: fixed(r["gamma"], 3)),
        pair("sigma_sig", lambda r: fixed(r["sigma_sig"], 4)),
        pair("FALCONws correction iterations", lambda r: fixed(r["falconws_same_point_trials"], 2)),
        pair("FALCON++ total trials upper estimate", lambda r: fixed(r["trials_upper"], 2, "upper")),
    ], "Both samplers are evaluated at the same FALCON++ narrow-width parameters; the FALCONws formula is recomputed.")

    all_sizes = list(literature) + [{
        "scheme": "FALCON++", "n": r["n"], "q": r["q"],
        "pk_bytes": r["public_key_payload_bytes"], "sig_bytes": sizes[r["n"]],
        "classical_bits": min(r["key_recovery"]["classical"], r["forgery"]["classical"]),
        "quantum_bits": min(r["key_recovery"]["quantum"], r["forgery"]["quantum"]),
    } for r in computed]
    size_note = "FALCON++ values are recomputed; FALCON/FALCON+ and FALCONws rows are cited literature inputs in data/literature_baselines.json. Signature sizes are entropy estimates, not measured padded-wire sizes."
    rows = []
    for scheme in ["FALCON / FALCON+", "FALCONws", "FALCON++"]:
        row = [scheme]
        for n in [512, 1024]:
            r = next(x for x in all_sizes if x["scheme"] == scheme and x["n"] == n)
            row += [r["pk_bytes"], r["sig_bytes"], f'{r["classical_bits"]}/{r["quantum_bits"]}']
        rows.append(row)
    write(2, "Sizes and heuristic attack costs at each scheme's own parameters",
          ["Scheme", "512 pk (B)", "512 sig (B)", "512 C/Q", "1024 pk (B)", "1024 sig (B)", "1024 C/Q"], rows, size_note)

    from falconpp.parameters import FALCONPP_I_1245, FALCONPP_1024_117, SALT_BITS
    parameters = {r.n: r for r in [FALCONPP_I_1245, FALCONPP_1024_117]}
    write(3, "Selected parameters", header, [
        pair("Target security lambda", lambda r: parameters[r["n"]].target_security),
        pair("q", lambda r: r["q"]),
        pair("gamma", lambda r: fixed(r["gamma"], 3)),
        pair("sigma_sig", lambda r: fixed(r["sigma_sig"], 4)),
        pair("beta", lambda r: r["beta"]),
        pair("Sampler-loss budget", lambda r: f'{parameters[r["n"]].sampler_loss_bits} bits'),
        pair("Salt wire length / entropy", lambda r: f'{parameters[r["n"]].salt_bytes} bytes / {SALT_BITS} bits'),
    ], "Inputs are read from the selected parameter registry. Target security and the loss budget are design inputs.")

    ordered = sorted(all_sizes, key=lambda r: (r["n"], ["FALCON / FALCON+", "FALCONws", "FALCON++"].index(r["scheme"])))
    write(4, "Parameter and size comparison", ["Scheme", "n", "q", "pk (B)", "sig (B)"],
          [[r["scheme"], r["n"], r["q"], r["pk_bytes"], r["sig_bytes"]] for r in ordered], size_note)

    def epsilon(r):
        x = mp.mpf(r["r_infinity_minus_one_upper"])
        exponent = int(mp.floor(mp.log10(x)))
        mantissa = fixed(x / mp.power(10, exponent), 2, "upper")
        return f"1+{mantissa}e{exponent}"

    write(5, "Sampling parameters and heuristic-profile bounds", header, [
        pair("Moment order k", lambda r: r["moment_order"]),
        pair("Global Gamma_hat (display only)", lambda r: fixed(r["gamma_hat"], 6)),
        pair("r_infinity upper estimate", epsilon),
        pair("p_corr lower estimate", lambda r: fixed(r["p_corr_lower"], 6, "lower")),
        pair("p_norm lower estimate", lambda r: f'1-exp(-{fixed(r["norm_tail_exponent"], 4, "lower")})'),
        pair("FALCON++ total trials", lambda r: fixed(r["trials_upper"], 2, "upper")),
        pair("FALCONws correction trials", lambda r: fixed(r["falconws_same_point_trials"], 2)),
    ], "Determinant-normalized heuristic profile, evaluated at the registered orders 130/145. Upper/lower bounds use outward display rounding; Gamma_hat is display-only.")

    write(6, "Heuristic costs and chi-BDD model diagnostics", header, [
        pair("Key recovery BKZ block size", lambda r: r["key_recovery"]["b"]),
        pair("Key recovery C/Q", lambda r: f'{r["key_recovery"]["classical"]}/{r["key_recovery"]["quantum"]}'),
        pair("Forgery BKZ block size", lambda r: r["forgery"]["b"]),
        pair("Forgery C/Q", lambda r: f'{r["forgery"]["classical"]}/{r["forgery"]["quantum"]}'),
        pair("chi-BDD BKZ block size", lambda r: r["chi_bdd_generic_diagnostic"]["b"]),
        pair("chi-BDD classical reduction exponent", lambda r: int(mp.floor(mp.mpf(r["chi_bdd_generic_diagnostic"]["classical_reduction_exponent"])))),
        pair("chi-BDD log2 sufficient samples", lambda r: r["chi_bdd_generic_diagnostic"]["table_integer_samples"]),
    ], "Numerical cost models only. The single-character chi-BDD model omits the preimage oracle. Sample exponents 1401/2813 are not security bits; displayed integer exponents follow the manuscript's floor convention.")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported Tables 1-6 (CSV + Markdown) to {out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "implement"))
    export_tables()
