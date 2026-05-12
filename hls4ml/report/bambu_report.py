import glob
import os
import xml.etree.ElementTree as ET
from hls4ml.report.vivado_report import _parse_power_report, _parse_implementation_report, _parse_timing_report, _parse_csim_results, _parse_rtl_cosim_results


def _coerce_value(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return raw
    try:
        return int(raw)
    except (ValueError, TypeError):
        try:
            return float(raw)
        except (ValueError, TypeError):
            return raw


def _parse_result_file(path):
    tree = ET.parse(path)
    root = tree.getroot()

    meta = {
        'Args':      root.attrib.get('args'),
        'Version':   root.attrib.get('version'),
        'Timestamp': root.attrib.get('timestamp'),
        'Benchmark': root.attrib.get('benchmark'),
        'File':      os.path.basename(path),
    }

    metrics = {}

    # --- Resource metrics (REGISTERS, SLACK, LUTS, etc.) ---
    resources = root.find('resources')
    if resources is not None:
        for key, val in resources.attrib.items():
            metrics[key] = _coerce_value(val)

    # --- Timing / simulation metrics ---
    # <timing><evaluation return_value="0"><run>X</run></evaluation></timing>
    timing = root.find('timing')
    if timing is not None:
        evaluation = timing.find('evaluation')
        if evaluation is not None:
            runs = [_coerce_value(r.text) for r in evaluation.findall('run')]
            if runs:
                metrics['Total cycles']         = sum(runs)
                metrics['Number of executions'] = len(runs)
                metrics['Average execution']    = sum(runs) / len(runs)

    return {'meta': meta, 'metrics': metrics}


def parse_bambu_report(hls_dir, part_family):
    """Parse bambu_results XML files from ``hls_dir``.
    If target is from Xilinx, parse Vivado reports.
    Must be extended to parse reports from differing manufacturers.

    Returns a dictionary with the parsed entries.
    """
    result = {}

    # Parse CSim and Cosim
    csim_results = _parse_csim_results(hls_dir)
    if csim_results is not None:
        result['CSimResults'] = csim_results

    cosim_results = _parse_rtl_cosim_results(hls_dir)
    if cosim_results is not None:
        result['CosimResults'] = cosim_results

    # Parse metrics reported by Bambu
    pattern = os.path.join(hls_dir, 'bambu_results*.xml')
    matches = sorted(glob.glob(pattern))
    if matches:
        parsed = [_parse_result_file(path) for path in matches]
        result['BambuMetrics'] = parsed[-1]['metrics']

    # Parse Vivado reports if target is from Xilinx
    if part_family == "Xilinx":
        implementation_report = _parse_implementation_report(hls_dir, is_vivado_accelerator=False, percentage_columns=False)
        if implementation_report is not None:
            result['ImplementationReport'] = implementation_report
        else:
            print('Implementation report not found.')

        timing_report = _parse_timing_report(hls_dir, is_vivado_accelerator=False)
        if timing_report is not None:
            result['TimingReport'] = timing_report
        else:
            print('Timing report not found.')

        power_report = _parse_power_report(hls_dir, is_vivado_accelerator=False)
        if power_report is not None:
            result['PowerReport'] = power_report
        else:
            print('Power report not found.')

    return result
