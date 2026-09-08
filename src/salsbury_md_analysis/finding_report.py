"""Evidence-first reader reports; presentation never prunes scientific outputs."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import quote


CONTEXT_FIELDS = {"title", "question", "systems", "comparisons", "population",
                  "weighting", "primary_selection", "methods", "figure_captions", "finding_context"}
FINDING_CONTEXT_FIELDS = {"population", "weighting", "primary_selection", "uncertainty",
                          "structural_artifact_ids"}


def validate_scientific_context(context):
    """Validate author-supplied descriptions without guessing chemistry or controls."""
    if not isinstance(context, dict) or set(context) - CONTEXT_FIELDS:
        raise ValueError("reporting.scientific_context has unknown fields or is not an object")
    for key in CONTEXT_FIELDS - {"systems", "comparisons", "methods", "figure_captions", "finding_context"}:
        if key in context and not isinstance(context[key], str):
            raise ValueError(f"scientific_context.{key} must be text")
    for key in ("comparisons", "methods"):
        if key in context and (not isinstance(context[key], list) or
                               not all(isinstance(item, str) for item in context[key])):
            raise ValueError(f"scientific_context.{key} must be a list of text")
    systems = context.get("systems", {})
    if not isinstance(systems, dict):
        raise ValueError("scientific_context.systems must map system IDs to descriptions")
    for sid, value in systems.items():
        if (not isinstance(sid, str) or not sid or not isinstance(value, dict)
                or set(value) - {"label", "description"}
                or not all(isinstance(item, str) for item in value.values())):
            raise ValueError("each scientific_context system accepts label and description text")
    captions = context.get("figure_captions", {})
    if not isinstance(captions, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in captions.items()
    ):
        raise ValueError("scientific_context.figure_captions must map artifact IDs to text")
    notes = context.get("finding_context", {})
    if not isinstance(notes, dict):
        raise ValueError("scientific_context.finding_context must map finding IDs to objects")
    for fid, note in notes.items():
        if (not isinstance(fid, str) or not isinstance(note, dict)
                or set(note) - FINDING_CONTEXT_FIELDS):
            raise ValueError("invalid scientific_context.finding_context entry")
        for key, value in note.items():
            if key == "structural_artifact_ids":
                if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                    raise ValueError("structural_artifact_ids must be a list of artifact IDs")
            elif not isinstance(value, str):
                raise ValueError(f"finding_context.{fid}.{key} must be text")
    return context


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md(value):
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _name(value):
    names = {"global_common_heavy": "Whole complex", "macromolecular_trace": "Polymer trace",
             "protein_dna_interface": "Protein–DNA interface"}
    return names.get(str(value), str(value).replace("_", " "))


def _statement(row, labels=None):
    # Only presentation boilerplate is removed. The recorded statement, numbers,
    # signs, denominators, ranking, and evidence level remain unchanged in JSON/CSV.
    value = str(row.get("statement", ""))
    value = value.replace("differs observationally", "differs").replace("differs descriptively", "differs").replace("Largest observational ", "Largest ")
    value = value.replace("Largest descriptive ", "Largest ")
    for sid, label in sorted((labels or {}).items(), key=lambda item: -len(item[0])):
        if len(sid) >= 3:
            value = re.sub(r"(?<![\w])" + re.escape(sid) + r"(?![\w])", lambda _: label, value)
    for internal in ("global_common_heavy", "macromolecular_trace", "protein_dna_interface"):
        value = value.replace(internal, _name(internal))
    value = value.replace("ion_coordination_geometry", "ion-coordination geometry")
    value = value.replace("SASA", "solvent-accessible area").replace("DCCM", "displacement-correlation")
    return value.replace("_", " ")


STYLE = """
body{font:17px/1.55 system-ui,sans-serif;color:#000;margin:32px auto;max-width:1120px;padding:0 24px}
h1{font-size:2rem;border-bottom:4px solid #9E7E38;padding-bottom:14px;line-height:1.2}
h2{font-size:1.45rem;line-height:1.3}h3{font-size:1.12rem}a{color:#705612;text-underline-offset:3px}
nav{background:#f4f0e7;padding:16px;display:flex;flex-wrap:wrap;gap:16px}
section{margin:36px 0}article{margin:36px 0 48px;break-inside:avoid}
.scope,figcaption{color:#53565A}figure{margin:20px 0}img{display:block;max-width:100%;height:auto}
figcaption{margin:10px 0;font-size:0.94rem}.table-scroll{overflow:auto}
table{border-collapse:collapse;width:100%;margin:18px 0}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd;vertical-align:top}
th{background:#CEB888;color:#000}details{margin:20px 0}summary{cursor:pointer;font-weight:600}
.finding-number{color:#9E7E38;font-weight:700}.warning{border-left:3px solid #9C2F2F;padding-left:12px}
@media print{nav{display:none}body{max-width:none;font-size:11pt}a{color:#000}img{max-height:7in}details{display:block}}
"""


def _page(title, content):
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{STYLE}</style></head>'
            f'<body><h1>{html.escape(title)}</h1>{content}</body></html>\n')


def write_finding_reader_reports(root: Path, findings: dict, context=None):
    """Write a short summary and an exhaustive evidence index without editing inputs.

    This renderer consumes the picker snapshot and the small artifact manifest,
    not the large trajectory-analysis JSON arrays. No network, plotting, model
    fitting, trajectory analysis, or scientific selection is performed here.
    """
    root = Path(root).resolve()
    context = validate_scientific_context(context or {})
    escape = html.escape
    manifest_path = root / "presentation-artifacts/presentation-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    artifacts = manifest.get("artifacts", [])
    by_id, verified, gaps = {}, {}, []
    for artifact in artifacts:
        aid = artifact["artifact_id"]
        if aid in by_id:
            raise ValueError(f"duplicate presentation artifact ID: {aid}")
        by_id[aid] = artifact
        relative = Path(str(artifact.get("relative_path", "")))
        path = root / "presentation-artifacts" / relative
        safe = bool(relative.parts) and not relative.is_absolute() and ".." not in relative.parts
        safe = safe and path.resolve().is_relative_to(root / "presentation-artifacts")
        if (not safe or not path.is_file() or
                not isinstance(artifact.get("artifact_sha256"), str) or
                _sha(path) != artifact["artifact_sha256"]):
            gaps.append({"artifact_id": aid, "issue": "missing, unsafe, or hash-mismatched artifact"})
            continue
        verified[aid] = quote(path.relative_to(root).as_posix())

    systems = set(context.get("systems", {}))
    for row in findings.get("all_candidates", []):
        systems.update(map(str, row.get("system_ids", [])))
    # Read only identity metadata; coordinate arrays and raw reports are untouched.
    system_path = root / "system.json"
    if system_path.is_file():
        raw_systems = json.loads(system_path.read_text()).get("systems", [])
        if isinstance(raw_systems, list):
            systems.update(str(item["system_id"]) for item in raw_systems
                           if isinstance(item, dict) and item.get("system_id"))
    labels = {sid: context.get("systems", {}).get(sid, {}).get("label", _name(sid))
              for sid in systems}
    candidate_ids = {str(row.get("finding_id")) for row in findings.get("all_candidates", [])}
    for fid in context.get("finding_context", {}):
        if fid not in candidate_ids:
            gaps.append({"finding_id": fid, "issue": "context refers to a finding absent from this snapshot"})
    for key in ("question", "population", "weighting", "primary_selection"):
        if not context.get(key):
            gaps.append({"context_field": key, "issue": "report context was not supplied; no value was inferred"})
    title = context.get("title") or "Molecular ensemble findings"
    nav = ('<nav aria-label="Report sections"><a href="prioritized_findings.html">Key findings</a>'
           '<a href="prioritized_findings_secondary.html">Secondary findings</a>'
           '<a href="finding_evidence.html">All figures, tables and data</a>'
           '<a href="prioritized_findings_qc.md">QC</a>'
           '<a href="prioritized_findings_details.md">Methods and accounting</a></nav>')
    intro, md = [], [f"# {_md(title)}", ""]
    if context.get("question"):
        intro.append(f'<p>{escape(context["question"])}</p>')
        md.extend([context["question"], ""])
    if systems:
        intro.append('<section><h2>Systems and comparisons</h2><table><tr><th>System</th><th>Composition or condition</th></tr>')
        md.extend(["## Systems and comparisons", "", "| System | Composition or condition |", "|---|---|"])
        ordered_systems = list(context.get("systems", {})) + sorted(systems - set(context.get("systems", {})))
        for sid in ordered_systems:
            definition = context.get("systems", {}).get(sid, {}).get("description", "")
            intro.append(f'<tr><td>{escape(labels[sid])}</td><td>{escape(definition)}</td></tr>')
            md.append(f"| {_md(labels[sid])} | {_md(definition)} |")
            if not definition:
                gaps.append({"system_id": sid, "issue": "system composition/condition was not supplied"})
        intro.append('</table></section>')
        md.append("")
    for value in context.get("comparisons", []):
        intro.append(f'<p>{escape(value)}</p>')
        md.extend([value, ""])
    for key, label in (("population", "Population"), ("weighting", "Pooling"),
                       ("primary_selection", "Primary selection")):
        if context.get(key):
            intro.append(f'<p><strong>{label}:</strong> {escape(context[key])}</p>')
            md.extend([f"{label}: {context[key]}", ""])

    methods_html = ""
    if context.get("methods"):
        methods_html = '<section><h2>Scientific methods</h2>' + ''.join(
            f'<p>{escape(item)}</p>' for item in context["methods"]) + '</section>'

    figure_numbers, table_numbers = {}, {}

    def artifact_links(row):
        links = []
        for ref in row.get("presentation_artifacts", []):
            aid = ref.get("artifact_id")
            if aid in verified:
                artifact = by_id[aid]
                label = f'{artifact["artifact_type"].capitalize()}: {artifact["title"]}'
                links.append((label, verified[aid]))
        return list(dict.fromkeys(links))

    def render_rows(rows, start, inline):
        cards, prose = [], []
        for rank, row in enumerate(rows, start=start):
            statement = _statement(row, labels)
            fid = str(row.get("finding_id", rank))
            note = context.get("finding_context", {}).get(fid, {})
            anchor = "finding-" + quote(fid, safe="")
            cards.append(f'<article id="{escape(anchor)}"><h2><span class="finding-number">{rank}.</span> {escape(statement)}</h2>')
            prose.extend([f"### {rank}. {_md(statement)}", ""])
            scope = " versus ".join(labels.get(str(sid), _name(sid)) for sid in row.get("system_ids", []))
            if scope:
                cards.append(f'<p class="scope">{escape(scope)}</p>')
            p = row.get("adjusted_p_value")
            if isinstance(p, (int, float)):
                text = f"Benjamini–Hochberg adjusted p = {p:.4g}."
                cards.append(f'<p>{text}</p>'); prose.extend([text, ""])
            for key, label in (("population", "Population"), ("weighting", "Pooling"),
                               ("primary_selection", "Selection"), ("uncertainty", "Uncertainty")):
                if note.get(key):
                    cards.append(f'<p><strong>{label}:</strong> {escape(note[key])}</p>')
                    prose.extend([f"{label}: {note[key]}", ""])
            # Only exact finding-target matches can appear as claim-supporting
            # panels. Broad report-level matches remain available as links.
            refs = list(row.get("presentation_artifacts", []))
            structural_ids = note.get("structural_artifact_ids", [])
            for aid in structural_ids:
                if aid not in verified:
                    gaps.append({"finding_id": fid, "artifact_id": aid,
                                 "issue": "requested structural evidence is not a verified artifact"})
                elif aid not in {ref.get("artifact_id") for ref in refs}:
                    refs.append({"artifact_id": aid})
            exact = row.get("presentation_artifact_match") == "exact_target"
            if inline:
                # A statistical plot is not a structural image. Explicit IDs or
                # manifest purpose identify coordinate-derived panels; never
                # infer that role from a caption mentioning a molecule.
                structures = [by_id[ref["artifact_id"]] for ref in refs
                              if ref.get("artifact_id") in verified and
                              (exact or ref.get("artifact_id") in structural_ids)]
                requires_structure = (row.get("category") in {"fes", "clustering", "conformation", "coupled_interaction"}
                                      or row.get("module_id") in {"pca_fes_basins", "clustering_kmeans", "dccm", "pooled_rmsf"})
                if requires_structure:
                    has_coordinates = any(a.get("artifact_type") == "structure" for a in structures)
                    has_panel = any(a.get("artifact_type") == "figure" and
                                    (a["artifact_id"] in structural_ids or
                                     a.get("purpose") in {"structural_figure", "representative_structure_figure"})
                                    for a in structures)
                    if not has_coordinates or not has_panel:
                        gaps.append({"finding_id": fid, "issue": "structural claim needs a coordinate-derived panel and its sampled structure",
                                     "coordinates_linked": has_coordinates, "structural_panel_linked": has_panel})
                if not structures:
                    gaps.append({"finding_id": fid, "issue": "no exact finding-target evidence was verified"})
            shown = 0
            for ref in refs:
                aid = ref.get("artifact_id")
                artifact = by_id.get(aid, {})
                if not inline or not (exact or aid in structural_ids) or aid not in verified or artifact.get("artifact_type") != "figure":
                    continue
                if Path(artifact["relative_path"]).suffix.lower() == ".svg":
                    with (root / "presentation-artifacts" / artifact["relative_path"]).open(encoding="utf-8") as handle:
                        header = handle.read(32768)
                    if "Reported effect (see table for units)" in header:
                        # Legacy scalar-effect bars add no distribution or spatial
                        # information. Show the numbers; keep the figure in the index.
                        continue
                if shown >= 2:
                    break
                if aid in figure_numbers:
                    number = figure_numbers[aid]
                    cards.append(f'<p>See <a href="#figure-{number}">Figure {number}</a>.</p>')
                    continue
                shown += 1
                number = len(figure_numbers) + 1
                figure_numbers[aid] = number
                caption = context.get("figure_captions", {}).get(aid) or artifact.get("caption") or artifact["title"]
                if not (context.get("figure_captions", {}).get(aid) or artifact.get("caption")):
                    gaps.append({"artifact_id": aid, "issue": "caption uses source title; detailed caption not supplied"})
                caption_text = f"Figure {number}. {caption}"
                href = verified[aid]
                cards.append(f'<figure id="figure-{number}"><a href="{href}"><img loading="lazy" src="{href}" alt="{escape(artifact["title"])}"></a><figcaption>{escape(caption_text)}</figcaption></figure>')
                prose.extend([f"![{_md(artifact['title'])}]({href})", "", caption_text, ""])
            if inline and exact:
                for ref in refs:
                    aid = ref.get("artifact_id")
                    artifact = by_id.get(aid, {})
                    if aid not in verified or artifact.get("artifact_type") != "table":
                        continue
                    if aid in table_numbers:
                        cards.append(f'<p>See <a href="#table-{table_numbers[aid]}">Table {table_numbers[aid]}</a>.</p>')
                        break
                    path = root / "presentation-artifacts" / artifact["relative_path"]
                    if path.suffix.lower() not in {".csv", ".tsv"}:
                        continue
                    try:
                        with path.open(encoding="utf-8", newline="") as handle:
                            reader = csv.reader(handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
                            # Bounded preview only. The complete numerical table is linked.
                            from itertools import islice
                            preview = list(islice(reader, 10))
                    except (UnicodeError, csv.Error):
                        continue
                    if not preview or max(map(len, preview)) > 12:
                        continue
                    excluded = {"finding_number", "finding_label", "comparison_family", "absolute_effect_value", "finding"}
                    if len(preview) > 2 and "finding" in preview[0]:
                        # Multi-row comparisons must keep the molecular/state
                        # identity carried by each statement, not anonymous effects.
                        excluded.discard("finding")
                        excluded.update({"left_system_id", "right_system_id"})
                    columns = [i for i, key in enumerate(preview[0]) if key not in excluded
                               and any(i < len(values) and values[i] for values in preview[1:])]
                    if not columns or len(columns) > 6:
                        continue
                    number = len(table_numbers) + 1
                    table_numbers[aid] = number
                    caption = f'Table {number}. {artifact["title"]}'
                    cards.append(f'<div id="table-{number}" class="table-scroll"><table><caption>{escape(caption)}</caption>')
                    for i, values in enumerate(preview[:9]):
                        tag = "th" if i == 0 else "td"
                        cells = []
                        for column in columns:
                            value = values[column] if column < len(values) else ""
                            if i == 0:
                                key = value
                                value = {"left_system_id": "First system", "right_system_id": "Second system"}.get(key, _name(key).capitalize())
                                if key == "effect_value":
                                    module = artifact.get("module_id")
                                    value = {"dccm": "Correlation difference", "dihedral_distributions": "Difference (degrees)",
                                             "hydrogen_bond_discovery": "Occupancy difference (fraction)",
                                             "pca_fes_basins": "Population difference (fraction)",
                                             "solvent_accessible_surface_area": "Difference (Å²)",
                                             "pooled_rmsf": "Difference (Å)",
                                             "radial_distribution_functions": "Difference in g(r)"}.get(module, value)
                                    if all("standardized replica-mean difference" in " ".join(values) for values in preview[1:]):
                                        value = "Standardized replica-mean difference"
                            elif value in labels:
                                value = labels[value]
                            elif preview[0][column] == "finding":
                                value = _statement({"statement": value}, labels)
                            elif preview[0][column] not in {"system_id", "left_system_id", "right_system_id"}:
                                try:
                                    if "." in value or "e" in value.lower():
                                        value = f"{float(value):.5g}"
                                except ValueError:
                                    pass
                            cells.append(f'<{tag}>{escape(value)}</{tag}>')
                        cards.append('<tr>' + ''.join(cells) + '</tr>')
                    cards.append('</table></div>')
                    if len(preview) > 9:
                        cards.append(f'<p>First 8 rows shown. <a href="{verified[aid]}">Download the complete table</a>.</p>')
                    prose.extend([f"[{_md(caption)}]({verified[aid]})", ""])
                    break
            links = artifact_links({"presentation_artifacts": refs})
            if links:
                cards.append('<p>' + ' · '.join(f'<a href="{href}">{escape(label)}</a>' for label, href in links) + '</p>')
                prose.extend([' · '.join(f'[{_md(label)}]({href})' for label, href in links), ""])
            else:
                gaps.append({"finding_id": fid, "issue": "no verified figure, table, or structure link"})
            cards.append('</article>')
        return "".join(cards), prose

    headline_html, headline_md = render_rows(findings.get("headline_findings", []), 1, True)
    if not findings.get("headline_findings"):
        headline_html = '<p>No candidates met the configured headline rule. The complete candidate list remains in the evidence index.</p>'
    secondary_html, secondary_md = render_rows(findings.get("secondary_findings", []),
                                                len(findings.get("headline_findings", [])) + 1, False)
    md.extend(["## Key findings", "", *headline_md,
               "[Secondary findings](prioritized_findings_secondary.md) · "
               "[All figures, tables and data](finding_evidence.html) · "
               "[QC](prioritized_findings_qc.md) · "
               "[Methods and accounting](prioritized_findings_details.md)", ""])
    if context.get("methods"):
        md.extend(["## Scientific methods", ""])
        for item in context["methods"]:
            md.extend([item, ""])
    files = {
        "prioritized_findings.html": _page(title, nav + "".join(intro) + '<section><h2>Key findings</h2>' + headline_html + '</section>' + methods_html),
        "prioritized_findings.md": "\n".join(md),
        "prioritized_findings_secondary.html": _page("Secondary findings", nav + secondary_html),
        "prioritized_findings_secondary.md": "\n".join(["# Secondary findings", "", *secondary_md]),
    }
    index_rows, inventory = [], []
    for artifact in artifacts:
        aid = artifact["artifact_id"]
        href = verified.get(aid)
        title_cell = escape(artifact["title"])
        if href:
            title_cell = f'<a href="{href}">{title_cell}</a>'
        index_rows.append(f'<tr><td>{escape(_name(artifact.get("analysis_class", "")))}</td><td>{escape(artifact["artifact_type"])}</td><td>{title_cell}</td></tr>')
        inventory.append({"artifact_id": aid, "analysis_class": artifact.get("analysis_class", ""),
                          "artifact_type": artifact["artifact_type"], "title": artifact["title"],
                          "relative_path": f'presentation-artifacts/{artifact["relative_path"]}',
                          "sha256": artifact.get("artifact_sha256", ""), "verified": bool(href),
                          "source_reports": ";".join(artifact.get("source_report_paths", []))})
    files["finding_evidence.html"] = _page("All figures, tables and data", nav +
        f'<p>{len(artifacts)} artifacts; {len(findings.get("all_candidates", []))} candidate findings.</p>' +
        '<p><a href="prioritized_findings.csv">Every candidate (CSV)</a> · <a href="prioritized_findings.json">Every candidate and provenance (JSON)</a> · <a href="finding_evidence.csv">Artifact index (CSV)</a></p>' +
        '<p>Analysis files remain in their original locations. The summary does not remove figures, tables, numerical data, structures, alternative methods, or candidate findings.</p>' +
        '<div class="table-scroll"><table><tr><th>Analysis class</th><th>Type</th><th>Evidence</th></tr>' + "".join(index_rows) + '</table></div>')
    review = ["# Reader report review", "",
              "These checks identify missing reporting evidence. They do not change results, ranks, or the supporting archive.", "",
              "| Finding, artifact, or field | Required review |", "|---|---|"]
    for gap in gaps:
        identity = " / ".join(str(gap[key]) for key in ("finding_id", "artifact_id", "system_id", "context_field") if gap.get(key))
        review.append(f"| {_md(identity)} | {_md(gap['issue'])} |")
    if not gaps:
        review = review[:4] + ["No automated evidence-link or supplied-context gaps were found. Visual and scientific review remain separate checks.", ""]
    files["finding_reader_review.md"] = "\n".join(review) + "\n"
    for name, content in files.items():
        (root / name).write_text(content, encoding="utf-8")
    fields = ["artifact_id", "analysis_class", "artifact_type", "title", "relative_path", "sha256", "verified", "source_reports"]
    with (root / "finding_evidence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(inventory)
    checks = {"report_schema": "evidence-first-reader-report-v1", "artifact_count": len(artifacts),
              "verified_artifact_count": len(verified), "candidate_count": len(findings.get("all_candidates", [])),
              "source_data_modified": False, "ranking_modified": False, "review_items": gaps,
              "human_review_path": "finding_reader_review.md",
              "archive_policy": "selective_opening_complete_supporting_archive",
              "scientific_context": context,
              "source_manifest_sha256": _sha(manifest_path) if manifest_path.is_file() else None,
              "generated_files": {name: _sha(root / name) for name in [*files, "finding_evidence.csv"]}}
    checks_path = root / "finding_reader_report_checks.json"
    checks_path.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"html_path": str(root / "prioritized_findings.html"),
            "html_sha256": _sha(root / "prioritized_findings.html"),
            "evidence_index_path": str(root / "finding_evidence.html"),
            "reader_report_checks_path": str(checks_path),
            "reader_report_checks_sha256": _sha(checks_path)}
