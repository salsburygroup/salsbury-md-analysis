"""Presentation-only support for explicit abstentions and equal-time tensors."""
import html
import math


def _coskewness_color(value, bound):
    fraction = min(1.0, abs(value) / bound)
    endpoint = (83, 86, 90) if value < 0 else (158, 126, 56)
    rgb = tuple(round(255 + fraction * (c - 255)) for c in endpoint)
    return "#%02X%02X%02X" % rgb


def distribution_abstentions(destination, path, report, module, artifacts):
    from .presentation_artifacts import _register_pair, _report_context, _view_artifact_directory
    distributions = report.get('distribution_reports', [])
    rows = [{'metric_id': d['metric_id'], 'status': d['status'], 'reason': d['reason']}
            for d in distributions if isinstance(d, dict) and d.get('status') == 'not_estimable'
            and isinstance(d.get('metric_id'), str) and isinstance(d.get('reason'), str) and d['reason']]
    if not rows:
        return False
    context = _report_context(path, report)
    subject = context.get('system_id', context.get('view_id', 'All systems'))
    title = f'{subject}: geometry distribution availability'
    reasons = sorted(set(row['reason'] for row in rows))
    lines = [f'{len(rows)} of {len(distributions)} requested distributions are not estimable.',
             'The table lists every affected metric and the recorded reason.',
             'No distribution or uncertainty estimate has been substituted.']
    import textwrap
    for reason in reasons:
        lines += textwrap.wrap('Recorded reason: ' + reason, width=95)
    body = '<rect width="100%" height="100%" fill="white"/>'
    body += f'<text x="25" y="35" font-size="21">{html.escape(title)}</text>'
    for i, line in enumerate(lines):
        body += f'<text x="25" y="{75 + 25*i}" font-size="15">{html.escape(line)}</text>'
    _register_pair(destination, path, artifacts, module_id=module,
        purpose='distribution_availability', title=title,
        directory=_view_artifact_directory(destination, module, context) / 'availability',
        rows=rows, fieldnames=('metric_id', 'status', 'reason'),
        svg=(920, 100 + 25*len(lines), body),
        context={**context, 'availability_scope': 'distribution_reports',
                 'unavailable_metric_count': len(rows), 'total_metric_count': len(distributions)},
        primary_human_output=False)
    return len(rows) == len(distributions)


def tensor_values(report):
    value = report.get('analyses', {}).get('coskewness', {}).get('coskewness')
    if value is None:
        return []
    from .presentation_artifacts import PresentationArtifactError
    count = report.get('analyses', {}).get('coskewness', {}).get('feature_count')
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 or not isinstance(value, list) or len(value) != count:
        raise PresentationArtifactError('coskewness tensor feature dimension differs')
    numbers = []
    for plane in value:
        if not isinstance(plane, list) or len(plane) != count:
            raise PresentationArtifactError('coskewness tensor has an invalid plane')
        for row in plane:
            if not isinstance(row, list) or len(row) != count:
                raise PresentationArtifactError('coskewness tensor has an invalid row')
            for number in row:
                if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number):
                    raise PresentationArtifactError('coskewness tensor contains a nonfinite value')
                numbers.append(float(number))
    return numbers


def coskewness_artifacts(destination, path, report, artifacts, shared_bound):
    from .presentation_artifacts import (_register_pair, _report_context,
        _view_artifact_directory, human_label, PresentationArtifactError)
    numbers = tensor_values(report)
    if not numbers:
        return
    analysis = report['analyses']['coskewness']
    tensor = analysis['coskewness']
    n = len(tensor)
    components = report.get('settings', {}).get('component_indices')
    if (not isinstance(components, list) or len(components) != n
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in components)
            or len(set(components)) != n):
        raise PresentationArtifactError('coskewness feature identities are unavailable')
    prefix = 'PC' if report['settings'].get('feature_source') == 'common_pca' else 'Feature'
    labels = [f'{prefix} {i}' for i in components]
    context = _report_context(path, report)
    directory = _view_artifact_directory(destination, 'information_dynamics', context) / 'coskewness'
    bound = max(float(shared_bound), max(abs(x) for x in numbers), 1e-12)
    for k, plane in enumerate(tensor):
        title = f'Equal-time coskewness: {labels[k]} held fixed'
        subject = ' / '.join(human_label(context[x]) for x in ['system_id', 'view_id'] if x in context)
        subject += f"; {analysis.get('observation_count', 'unreported')} evaluated observations"
        body = f'<text class="title" x="24" y="30">{html.escape(title)}</text>'
        body += f'<text class="subtitle" x="24" y="54">{html.escape(subject)}</text>'
        left, top, side = 95, 85, 490
        cell = side / n
        rows = []
        for i, row in enumerate(plane):
            for j, value in enumerate(row):
                rows.append({'fixed_feature': labels[k], 'row_feature': labels[i],
                             'column_feature': labels[j], 'coskewness': value})
                color = _coskewness_color(value, bound)
                ink = 'white' if abs(value) > .65*bound else 'black'
                body += f'<rect x="{left+j*cell}" y="{top+i*cell}" width="{cell}" height="{cell}" fill="{color}"/>'
                if n <= 16:
                    body += f'<text x="{left+(j+.5)*cell}" y="{top+(i+.5)*cell+5}" text-anchor="middle" font-size="13" style="fill:{ink}">{value:.3g}</text>'
        for i, label in enumerate(labels):
            if i % max(1, math.ceil(n / 12)) and i != n - 1:
                continue
            body += f'<text class="small" x="{left+(i+.5)*cell}" y="{top+side+23}" text-anchor="middle">{html.escape(label)}</text>'
            body += f'<text class="small" x="{left-12}" y="{top+(i+.5)*cell+4}" text-anchor="end">{html.escape(label)}</text>'
        body += '<text class="small" x="24" y="640">Mean product of standardized feature triplets; dimensionless.</text>'
        body += '<text class="small" x="24" y="664">Equal-time statistic; no lagged kinetics or causal interpretation.</text>'
        body += f'<text class="small" x="610" y="105">Shared scale</text><text class="small" x="610" y="126">±{bound:.3g}</text>'
        for q in range(60):
            val = bound*(1-2*q/59)
            body += f'<rect x="624" y="{150+q*5}" width="26" height="6" fill="{_coskewness_color(val, bound)}"/>'
        body += f'<text class="small" x="660" y="160">+{bound:.3g}</text><text class="small" x="660" y="304">0</text><text class="small" x="660" y="450">−{bound:.3g}</text>'
        _register_pair(destination, path, artifacts, module_id='information_dynamics',
            purpose=f'coskewness_fixed_component_{components[k]}', title=title, directory=directory,
            rows=rows, fieldnames=('fixed_feature','row_feature','column_feature','coskewness'),
            svg=(820, 700, body), context={**context, 'fixed_component': components[k],
                'feature_source': report['settings'].get('feature_source'), 'statistic': 'equal_time_coskewness',
                'observation_count': analysis.get('observation_count'), 'shared_color_bound': bound})
