"""
Render manuscript-ready APA-style tables from current result CSVs.

Outputs are written to paper/tables/ as Markdown and LaTeX fragments. The
tables use current CSV results as the source of truth and avoid the stale
summary figures/report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "output" / "tables"
OUT_DIR = ROOT / "paper" / "tables"


def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.3f}"


def money(x: float) -> str:
    if pd.isna(x):
        return ""
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def pval(x: float) -> str:
    if pd.isna(x):
        return ""
    if x < 0.001:
        return "< .001"
    return f"{x:.3f}".replace("0.", ".")


def est_se(estimate: float, se: float, p: float | None = None, multiplier: float = 100.0) -> str:
    est = estimate * multiplier
    serr = se * multiplier
    return f"{est:.3f}{stars(p)} ({serr:.3f})"


def write_table(name: str, title: str, df: pd.DataFrame, note: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / f"{name}.md"
    tex_path = OUT_DIR / f"{name}.tex"

    header = "| " + " | ".join(df.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in df.astype(str).itertuples(index=False, name=None)
    ]
    markdown = [f"**{title}**", "", header, divider, *body, "", f"*Note.* {note}", ""]
    md_path.write_text("\n".join(markdown), encoding="utf-8")

    def esc(value: object) -> str:
        text = str(value)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    colspec = "l" * len(df.columns)
    latex_lines = [
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(esc(col) for col in df.columns) + r" \\",
        r"\midrule",
    ]
    for row in df.itertuples(index=False, name=None):
        latex_lines.append(" & ".join(esc(value) for value in row) + r" \\")
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    latex = "\n".join(latex_lines)
    tex = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{{title}}}",
        latex.replace("\\toprule", "\\toprule").replace("\\bottomrule", "\\bottomrule"),
        f"\\begin{{minipage}}{{0.95\\linewidth}}\\footnotesize\\emph{{Note.}} {note}\\end{{minipage}}",
        "\\end{table}",
        "",
    ]
    tex_path.write_text("\n".join(tex), encoding="utf-8")


def narrow_main() -> None:
    df = pd.read_csv(TABLE_DIR / "narrow" / "01_main_model.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "Period": r["period"].replace("_", " "),
                "Sample": r["sample"],
                "Outcome": "Customers" if r["outcome"] == "lcus" else "Spending",
                "Estimate (SE), %": est_se(r["estimate"], r["std.error"], r["p.value"]),
                "95% CI, %": f"[{r['ci_low95']*100:.3f}, {r['ci_hi95']*100:.3f}]",
                "p": pval(r["p.value"]),
                "N": f"{int(r['nobs']):,}",
            }
        )
    out = pd.DataFrame(rows)
    write_table(
        "table_01_narrow_twfe_main",
        "Table 1. Narrow TWFE estimates of EVCS exposure on nearby POI activity",
        out,
        "Entries are log-point coefficients converted to percent units. Standard errors are clustered by POI and shown in parentheses. * p < .05, ** p < .01, *** p < .001.",
    )


def narrow_monetary() -> None:
    df = pd.read_csv(TABLE_DIR / "narrow" / "06_monetary_impacts.csv")
    out = pd.DataFrame(
        {
            "Period": df["period"].str.replace("_", " ", regex=False),
            "Sample": df["sample"],
            "Annual spend per POI": df["M_per_poi_annual"].map(money),
            "Annual customers per POI": df["F_per_poi_annual"].map(lambda x: f"{x:,.1f}"),
            "Annual spend within 500m": df["M_all_annual"].map(money),
        }
    )
    write_table(
        "table_02_narrow_monetary",
        "Table 2. Narrow-replication monetary interpretation",
        out,
        "Monetary impacts apply the narrow TWFE spend coefficient to average pretreatment spending, average ports per EVCS, and the average number of nearby POIs used in the replication calculation.",
    )


def broad_cs_and_stacked() -> None:
    cs = pd.read_csv(TABLE_DIR / "broad" / "corrected_cs_main_summary.csv")
    stacked = pd.read_csv(TABLE_DIR / "main" / "broad_stacked_own_port_results.csv")
    rows = []
    for _, r in cs.iterrows():
        rows.append(
            {
                "Model": "CS timing",
                "Term": "ATT",
                "Outcome": "Customers" if r["outcome"] == "lcus" else "Spending",
                "Estimate (SE), %": f"{np.expm1(r['ATT'])*100:.3f} ({r['SE']*100:.3f})",
                "95% CI, %": f"[{np.expm1(r['CI_lower'])*100:.3f}, {np.expm1(r['CI_upper'])*100:.3f}]",
                "p": "",
                "N": "",
            }
        )
    labels = {
        "port_treat": "All ports",
        "port_treat_level2": "Level 2 ports",
        "port_treat_dc": "DC fast ports",
    }
    for _, r in stacked.iterrows():
        rows.append(
            {
                "Model": "Stacked own-port",
                "Term": labels[r["term"]],
                "Outcome": "Customers" if r["outcome"] == "lcus" else "Spending",
                "Estimate (SE), %": est_se(r["estimate"], r["std.error"], r["p.value"]),
                "95% CI, %": f"[{r['pct_ci_low95']:.3f}, {r['pct_ci_hi95']:.3f}]",
                "p": pval(r["p.value"]),
                "N": f"{int(r['nobs']):,}",
            }
        )
    out = pd.DataFrame(rows)
    write_table(
        "table_03_broad_cs_stacked",
        "Table 3. Broad replication estimates",
        out,
        "CS estimates are binary timing ATT estimates. Stacked estimates use a 6-month pre-window, 12-month post-window, stack-specific POI and county-month fixed effects, and POI-clustered standard errors.",
    )


def spatial_main_and_monetary() -> None:
    stacked = pd.read_csv(TABLE_DIR / "main" / "spatial_competition_stacked_results.csv")
    labels = {
        "competitor_port_treat": "All competitor ports",
        "competitor_port_treat_level2": "Level 2 competitor ports",
        "competitor_port_treat_dc": "DC fast competitor ports",
    }
    rows = []
    for _, r in stacked.iterrows():
        rows.append(
            {
                "Term": labels[r["term"]],
                "Outcome": "Customers" if r["outcome"] == "lcus" else "Spending",
                "Estimate (SE), %": est_se(r["estimate"], r["std.error"], r["p.value"]),
                "95% CI, %": f"[{r['pct_ci_low95']:.3f}, {r['pct_ci_hi95']:.3f}]",
                "p": pval(r["p.value"]),
                "N": f"{int(r['nobs']):,}",
            }
        )
    out = pd.DataFrame(rows)
    write_table(
        "table_04_spatial_stacked_main",
        "Table 4. Spatial competition stacked estimates",
        out,
        "The focal unit is a non-EVCS POI. Treatment is charger-port exposure at nearby same-sector competitors. Standard errors are clustered by POI.",
    )

    monetary = pd.read_csv(TABLE_DIR / "main" / "spatial_stacked_monetary_impacts.csv")
    money_rows = []
    for _, r in monetary.iterrows():
        money_rows.append(
            {
                "Effect": r["effect"],
                "Annual spend per competitor POI": money(r["annual_spend_effect_per_poi"]),
                "Aggregate annual spend": money(r["total_annual_effect_all_treated_pois"]),
                "Interpretation": r["interpretation"],
            }
        )
    write_table(
        "table_05_spatial_monetary",
        "Table 5. Monetary interpretation of spatial competition estimates",
        pd.DataFrame(money_rows),
        "Dollar values apply current stacked spatial spend coefficients to average active competitor-port exposure and baseline monthly spending among spatially treated focal competitor POIs.",
    )


def spatial_robustness() -> None:
    dist = pd.read_csv(TABLE_DIR / "robustness" / "stacked_spatial_distance_sensitivity.csv")
    dist = dist[dist["term"].isin(["competitor_port_treat", "competitor_port_treat_dc"])].copy()
    rows = []
    labels = {
        "competitor_port_treat": "All competitor ports",
        "competitor_port_treat_dc": "DC fast competitor ports",
    }
    for _, r in dist.iterrows():
        rows.append(
            {
                "Radius": f"{int(r['competition_radius_m'])}m",
                "Term": labels[r["term"]],
                "Outcome": "Customers" if r["outcome"] == "lcus" else "Spending",
                "Estimate (SE), %": est_se(r["estimate"], r["std.error"], r["p.value"]),
                "p": pval(r["p.value"]),
            }
        )
    write_table(
        "table_06_spatial_radius_robustness",
        "Table 6. Spatial competition radius robustness",
        pd.DataFrame(rows),
        "Radius robustness uses the stacked spatial competition estimator at competition radii above the 500m direct-treatment radius.",
    )


def poi_type() -> None:
    df = pd.read_csv(TABLE_DIR / "robustness" / "stacked_spatial_poi_type_heterogeneity.csv")
    df = df[df["term"].isin(["competitor_port_treat", "competitor_port_treat_dc"])].copy()
    rows = []
    labels = {
        "competitor_port_treat": "All competitor ports",
        "competitor_port_treat_dc": "DC fast competitor ports",
    }
    for _, r in df.iterrows():
        rows.append(
            {
                "POI type": r["poi_type"].replace("_", " "),
                "Term": labels[r["term"]],
                "Outcome": "Customers" if r["outcome"] == "lcus" else "Spending",
                "Estimate (SE), %": est_se(r["estimate"], r["std.error"], r["p.value"]),
                "p": pval(r["p.value"]),
            }
        )
    write_table(
        "table_07_spatial_poi_type",
        "Table 7. Spatial competition POI-type heterogeneity",
        pd.DataFrame(rows),
        "POI-type heterogeneity is estimated using the preferred stacked spatial competition design. Significant patterns are concentrated in restaurant exposure to DC fast competitor ports.",
    )


def pretrend() -> None:
    df = pd.read_csv(TABLE_DIR / "robustness" / "stacked_pretrend_placebo.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "Target": r["target"].title(),
                "Outcome": "Customers" if r["outcome"] == "lcus" else "Spending",
                "Term": r["term"].replace("_", " "),
                "Estimate": f"{r['estimate']:.4f}{stars(r['p.value'])}",
                "SE": f"{r['std.error']:.4f}",
                "p": pval(r["p.value"]),
            }
        )
    write_table(
        "table_08_stacked_pretrend_placebo",
        "Table 8. Stacked pretrend and placebo diagnostics",
        pd.DataFrame(rows),
        "Significant pre-period terms indicate residual timing-pattern differences. These diagnostics motivate cautious causal language for stacked estimates.",
    )


def main() -> None:
    narrow_main()
    narrow_monetary()
    broad_cs_and_stacked()
    spatial_main_and_monetary()
    spatial_robustness()
    poi_type()
    pretrend()


if __name__ == "__main__":
    main()
