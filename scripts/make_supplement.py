"""Supplementary material: plant parameters and every tuned gain, generated
from the same files the simulations read (Review 1: reproducibility)."""
import json
from pathlib import Path
R, P = Path("results"), Path("paper")
NAMES = {"lqr": "LQR", "mpc": "MPC", "fbl": "FBL+LQR", "smc": "SMC",
         "backstepping": "Backstepping", "pid": "PID"}
def fmt(v):
    m, e = f"{v:.4g}", None
    if "e" in m:
        m, e = m.split("e")
        return m + r"\times10^{" + str(int(e)) + "}"
    return m


def gains_table(path, caption):
    g = json.loads(Path(path).read_text())
    rows = [r"\begin{table}[h]\centering\scriptsize", rf"\caption{{{caption}}}",
            r"\begin{tabular}{@{}llp{0.52\columnwidth}@{}}", r"\toprule",
            r"Controller & Variant & Gains \\", r"\midrule"]
    for variant in ("classical", "momentum_aware", "integral"):
        for c, b in g.get(variant, {}).items():
            esc = lambda k: k.replace("_", r"\_")
            gs = ", ".join(r"\texttt{" + esc(k) + "}$=" + fmt(v) + "$" for k, v in b["gains"].items())
            extra = f"; $k_h={fmt(b['k_h'])}$" if variant == "momentum_aware" else (
                f"; $K_i={fmt(b['k_i'])}$" if variant == "integral" else "")
            if variant == "integral" and c == "pid":
                gs, extra = r"no separate arm: identical to the classical PID ($K_i$ unused)", ""
            rows.append(f"{NAMES[c]} & {variant.replace('_', ' ')} & {gs}{extra} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


body = [r"\documentclass[conference]{IEEEtran}", r"\usepackage{booktabs,amsmath}",
        r"\begin{document}",
        r"\title{Supplementary Material: Gravity-Torque Momentum Management for a Three-Wheel Reaction-Wheel Inverted Pendulum}",
        r"\author{\IEEEauthorblockN{Jose Luis Segundo Manayay, Javier Yanpier Garay Yovera, Francisco Jes\'us Rodr\'iguez Huiman}\IEEEauthorblockA{Universidad Nacional de Ingenier\'ia (UNI), Faculty of Mechanical Engineering, Lima, Peru}}",
        r"\maketitle",
        r"\section*{S1. Plant parameters (nominal design point)}",
        r"\begin{table}[h]\centering\footnotesize\input{table_params}\end{table}",
        r"The wheels are idealised discs whose rim reaches the seat distance ($r/d=1$); a real wheel of finite thickness would have to be smaller. "
        r"The Monte Carlo samples the fill ratio $r/d$ with $\pm20\%$ and caps it at 0.95, so most perturbed plants sit at the cap and have a smaller momentum budget than the nominal one.",
        r"\section*{S2. Tuned gains}",
        r"Gains of the paper (Table~I), tuned with the cost of Sec.~IV-H, and of the survival-first re-tuning reported in Sec.~VI-D. "
        r"LQR/MPC: state weights $q_\theta,q_\omega,q_h$ and input weight $r$; FBL+LQR: outer-loop weights; SMC: $\Lambda$, $K$, boundary layer $\phi$; "
        r"backstepping: $k_1,k_2$ ($\Gamma=I$, not tuned); PID: $k_p,k_i,k_d$. MPC: 25 steps of 40~ms, updates every 20~ms, OSQP. Momentum-management tilt clamp: 3$^\circ$.",
        gains_table(R / "tuned_gains.json", "Gains used in the paper"),
        gains_table(R / "tuned_gains_lexicographic.json", "Gains of the survival-first re-tuning"),
        r"\end{document}"]
(P / "supplement.tex").write_text("\n".join(body) + "\n")
print("wrote paper/supplement.tex")
