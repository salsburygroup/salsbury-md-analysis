"""Human-readable PaLD results, restricted to the reported sampled ensemble."""

from collections import Counter
import html
import math
import textwrap

from .presentation_artifacts import (
    PresentationArtifactError, _bar_svg, _finite, _histogram_svg,
    _register_pair, _report_context, _scott_histogram, _source,
    _view_artifact_directory, _write_csv, artifact_record, human_label,
)


LIMITATION = (
    "Fractions describe sampled observations only; they are not all-frame populations. "
    "Community IDs are local to this report. Cohesion does not establish kinetics, "
    "metastability, convergence or scientific acceptance."
)


def _integer(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise PresentationArtifactError(f"PaLD {name} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def _caption(svg, lines):
    width, height, body = svg
    lines = [part for line in lines for part in textwrap.wrap(line, max(40, (width-48)//7))]
    for index, line in enumerate(lines):
        body += f'<text class="small" x="24" y="{height + 20 + 20*index}">{html.escape(line)}</text>'
    return width, height + 28 + 20*len(lines), body


def pald_artifacts(output_root, path, report, artifacts):
    """Plot reported communities and sampled depth; never extrapolate labels."""
    module = "pald_community_analysis"
    n = _integer(report.get("sampled_observation_count"), "sampled_observation_count", positive=True)
    source_n = _integer(report.get("source_observation_count"), "source_observation_count", positive=True)
    observations = report.get("sampled_observations")
    communities = report.get("communities")
    if source_n < n or not isinstance(observations, list) or len(observations) != n:
        raise PresentationArtifactError("PaLD sampled observation count does not match its rows/source count")
    if not isinstance(communities, list) or not communities:
        raise PresentationArtifactError("PaLD community rows are required")
    counts = Counter()
    system_counts = Counter()
    system_communities = Counter()
    sample_indices = set()
    for row in observations:
        if not isinstance(row, dict) or not str(row.get("system_id", "")).strip():
            raise PresentationArtifactError("PaLD sampled observation requires system identity")
        cid = _integer(row.get("community_id"), "community_id", positive=True)
        index = _integer(row.get("pald_sample_index"), "pald_sample_index")
        if index in sample_indices or index >= n:
            raise PresentationArtifactError("PaLD sample indices must be unique and within the sampled count")
        sample_indices.add(index)
        counts[cid] += 1
        system_counts[str(row["system_id"])] += 1
        system_communities[str(row["system_id"]), cid] += 1
        for field in ("local_depth", "boundary_cohesion_fraction"):
            value = _finite(row.get(field))
            if value is None or value < 0 or (field == "boundary_cohesion_fraction" and value > 1):
                raise PresentationArtifactError(f"PaLD {field} contains an invalid value")
    seen = set()
    rows = []
    for community in communities:
        cid = _integer(community.get("community_id"), "community_id", positive=True)
        count = _integer(community.get("sampled_population"), "sampled_population", positive=True)
        fraction = _finite(community.get("sampled_population_fraction"))
        if cid in seen or counts[cid] != count or fraction is None or not math.isclose(fraction, count/n, abs_tol=1e-10):
            raise PresentationArtifactError("PaLD community membership/count/fraction does not match sampled observations")
        seen.add(cid)
        core = community.get("core_observation", {})
        rows.append({
            "community_id": cid, "sampled_observation_count": count,
            "sampled_denominator": n, "source_observation_count": source_n,
            "fraction_of_sampled_observations": count/n,
            "mean_local_depth": community.get("mean_local_depth"),
            **{f"core_{key}": core.get(key) for key in (
                "system_id", "replica_id", "segment_id", "member_id", "source_frame_index",
            )},
            "scope_limitation": LIMITATION,
        })
    if seen != set(counts):
        raise PresentationArtifactError("PaLD community rows omit sampled memberships")
    context = {**_report_context(path, report),
               "sampled_observation_count": n, "source_observation_count": source_n,
               "denominator_scope": "sampled_observations", "limitations": LIMITATION}
    directory = _view_artifact_directory(output_root, module, context)
    caption = [
        f"{context.get('system_id', 'Pooled systems')} | {human_label(context.get('view_id', 'pooled feature space'))}",
        f"Sample: {n:,} of {source_n:,} source feature observations ({100*n/source_n:.3g}%). Replicas are pooled.",
        f"Features: {report.get('settings', {}).get('feature_source', 'see source report')}; "
        f"components {report.get('settings', {}).get('component_indices', 'see source report')}; "
        f"standardized: {report.get('settings', {}).get('standardize_features', 'see source report')}.",
        "Fractions apply only to this sample; equivalent member observations are not independent replicas.",
        "Community IDs apply only within this report. No kinetic or scientific acceptance claim is made.",
        f"Strong-tie threshold: {report.get('strong_tie_threshold', 'see source report')}; half the mean cohesion diagonal.",
    ]
    title = "PaLD sampled community fractions"
    svg = _bar_svg(rows, title, "community_id", "fraction_of_sampled_observations", "Sampled observation fraction")
    svg = (svg[0], svg[1], svg[2] + '<text class="label" x="420" y="638" text-anchor="middle">Community ID</text>')
    _register_pair(output_root, path, artifacts, module_id=module, purpose="sampled_communities",
                   title=title, directory=directory, rows=rows, fieldnames=tuple(rows[0]),
                   svg=_caption(svg, [*caption, f"Figure shows first {min(60,len(rows))} of {len(rows)} communities; table contains all."]), context=context)

    # Each system contributes to the same pooled PaLD definition. Denominators
    # are its sampled observations, not all of its source trajectory frames.
    system_rows = [{"system_id": sid, "community_id": cid,
                    "sampled_observation_count": system_communities[sid, cid],
                    "system_sampled_denominator": denominator,
                    "fraction_of_system_sample": system_communities[sid, cid]/denominator,
                    "scope_limitation": LIMITATION}
                   for sid, denominator in sorted(system_counts.items()) for cid in sorted(seen)]

    def table(purpose, title, records):
        if not records:
            return
        target = directory / f"{purpose}.csv"
        fields = list(dict.fromkeys(key for row in records for key in row))
        _write_csv(target, fields, records)
        sources, hashes = _source(path)
        artifacts.append(artifact_record(
            artifact_type="table", module_id=module, purpose=purpose, title=title,
            relative_path=str(target.relative_to(output_root)), source_report_paths=sources,
            source_report_sha256=hashes, context=context, media_type="text/csv"))

    table("system_sampled_communities", "PaLD fractions within each system's sample", system_rows)
    table("sampled_observations", "PaLD sample identities, local depth and boundary cohesion", observations)
    ties = report.get("strongest_intercommunity_ties", [])
    table("reported_intercommunity_ties", "Reported strongest intercommunity ties (truncated by analysis settings)", ties)
    for field, label in (("local_depth", "Local depth (dimensionless)"),
                         ("boundary_cohesion_fraction", "Cross-community mutual cohesion fraction")):
        bins, settings = _scott_histogram([float(row[field]) for row in observations])
        # The generic helper has historical Angstrom keys; PaLD CSVs must not
        # claim distance units for dimensionless cohesion statistics.
        bins_out = [{key.replace("_angstrom", ""): value for key, value in row.items()} for row in bins]
        for row in bins_out:
            row.update(sampled_denominator=n, binning_rule=settings["rule"],
                       scope_limitation=LIMITATION)
        title = f"PaLD {field.replace('_', ' ')} distribution"
        svg = _histogram_svg(bins, title, label)
        svg = (svg[0], svg[1], svg[2].replace(">Frame fraction<", ">Sampled observation fraction<"))
        _register_pair(output_root, path, artifacts, module_id=module, purpose=field,
                       title=title, directory=directory, rows=bins_out,
                       fieldnames=tuple(bins_out[0]), svg=_caption(svg, [*caption,
                           "Scott's bin-width rule with the toolkit's 5–200-bin display bounds; constant data use one bin."]), context=context)
