"""Fair-tuning protocol: base gains first, then the augmentation gain.

Phase 1 searches each controller's own gains on the shared scenario set.
Phase 2 freezes them and searches only k_h (momentum management) or k_i
(integral action), so the variants of a controller differ by exactly one
number and the comparison isolates the augmentation.
"""

import concurrent.futures as cf
import time

from cubli.experiments import CONTROLLERS, SINGLE_VARIANT
from cubli.tuning import save, tune

VARIANTS = ["momentum_aware", "integral"]


def base_job(name):
    print(f"  [1] {name}", flush=True)
    return name, tune(name, "classical", verbose=False)


def aug_job(args):
    name, variant, gains = args
    print(f"  [2] {name} {variant}", flush=True)
    return name, variant, tune(name, variant, base_gains=gains, verbose=False)


if __name__ == "__main__":
    t0 = time.time()
    out = {"classical": {}, "momentum_aware": {}, "integral": {}}

    with cf.ProcessPoolExecutor(max_workers=len(CONTROLLERS)) as ex:
        for name, res in ex.map(base_job, CONTROLLERS):
            out["classical"][name] = res
            g = "  ".join(f"{k}={v:.4g}" for k, v in res["gains"].items())
            print(f"{name:<14}classical       cost={res['cost']:9.2f}  {g}", flush=True)

    jobs = [(n, v, out["classical"][n]["gains"])
            for n in CONTROLLERS for v in VARIANTS if n not in SINGLE_VARIANT]
    with cf.ProcessPoolExecutor(max_workers=min(len(jobs), 12)) as ex:
        for name, variant, res in ex.map(aug_job, jobs):
            out[variant][name] = res
            extra = res["k_h"] if variant == "momentum_aware" else res["k_i"]
            tag = "k_h" if variant == "momentum_aware" else "k_i"
            print(f"{name:<14}{variant:<16}cost={res['cost']:9.2f}  {tag}={extra:.4g}",
                  flush=True)

    save(out, "results/tuned_gains.json")
    print("")
    print(f"done in {time.time() - t0:.0f} s -> results/tuned_gains.json")
