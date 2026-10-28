"""python -m rna : component ageing table for the two illustrative records."""
from . import Turbine, assess, synthetic


def main():
    tb = Turbine()
    for kind, title in (("op", "Operating at 12 m/s"), ("storm", "Parked storm with wave slam")):
        print(f"\n{title}  (ageing: design-life hours used per hour)")
        for c in assess(synthetic(kind, tb), tb):
            a = "  —   " if c["ageing"] is None else f"{c['ageing']:8.3g}"
            chk = "" if c["check"] is None else f"   static/peak utilisation {c['check']:.2f}"
            print(f"  {c['name']:<30}{a}{chk}")
            for k, v in c["metrics"]:
                print(f"      {k:<38}{v}")


if __name__ == "__main__":
    main()
