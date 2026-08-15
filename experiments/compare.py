"""Entry point that pulls run metrics from the Augur-Elliptic MLflow
experiment and (re)produces the comparison table. Not a reimplementation --
this calls ablation/run_ablation.py's already-built, already-verified
build_table()/write_outputs() directly, so running this script regenerates
ablation/results/ablation_results.{csv,md} live from MLflow, not from the
static files.
"""

from ablation.run_ablation import build_table, write_outputs


def main():
    rows = build_table()
    csv_path, md_path = write_outputs(rows)

    print(f"Regenerated comparison table from the Augur-Elliptic MLflow experiment:")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    print()
    for r in rows:
        print(r)

    return rows


if __name__ == "__main__":
    main()
