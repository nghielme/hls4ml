import itertools
import os
from pathlib import Path

import yaml

"""
Create a Gitlab CI yml file with separate pytest jobs.

Files in BAMBU_SHARED_TESTS run once in the standard image and once in the
Bambu image. Pytest collection filters select the backend cases for each image.
"""

n_test_files_per_yml = int(os.environ.get('N_TESTS_PER_YAML', 4))

BAMBU_BACKENDS = ('Bambu', 'BambuAccelerator')
BAMBU_FILTER_ARGS = tuple(f'--backend-filter={backend}' for backend in BAMBU_BACKENDS)
BAMBU_EXCLUDE_ARGS = tuple(f'--backend-exclude={backend}' for backend in BAMBU_BACKENDS)

# Blacklisted tests will be skipped
BLACKLIST = {'test_reduction'}

# Long-running tests will not be bundled with other tests
LONGLIST = {'test_hgq_layers', 'test_hgq_players', 'test_qkeras', 'test_pytorch_api'}
KERAS3_LIST = {
    'test_keras_v3_api',
    'test_hgq2_mha',
    'test_einsum_dense',
    'test_qeinsum',
    'test_multiout_onnx',
    'test_keras_v3_profiling',
}

# Files containing both Bambu and non-Bambu backend parametrizations.
BAMBU_SHARED_TESTS = {
    'test_activations',
    'test_auto_precision',
    'test_build_bambu',
    'test_dense_unrolled',
    'test_keras_api',
    'test_multi_dense',
    'test_pooling',
    'test_softmax',
}

# Bambu-specific tests that are not expressed by backend parametrization.
BAMBU_ONLY_TESTS = {
    'test_report.py::test_bambu_report',
}

# Test files to split by individual test functions.
# Value = chunk size per CI job.
SPLIT_BY_TEST_CASE = {
    'test_keras_api': 1,
}


def batched(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, batch_size)):
        yield batch


def path_to_name(test_path):
    return Path(test_path).stem.replace('test_', '')


def uses_example_model(test_filename):
    with open(test_filename) as f:
        return 'example-models' in f.read()


def collect_test_functions_from_ast(test_file, test_root):
    import ast

    with open(test_file, encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=str(test_file))

    test_funcs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test'):
            test_funcs.append(f'{test_file.relative_to(test_root)}::{node.name}')
    return test_funcs


def pytest_template(path, bambu=False):
    keras3 = path.stem in KERAS3_LIST
    if bambu and keras3:
        return '.pytest-bambu-keras3-only'
    if bambu:
        return '.pytest-bambu'
    if keras3:
        return '.pytest-keras3-only'
    return '.pytest'


def job_name(test_paths, bambu=False):
    prefix = 'bambu-' if bambu else ''
    return prefix + '+'.join(path_to_name(path) for path in test_paths)


def test_case_job_name(path, nodeids, bambu=False):
    prefix = 'bambu-' if bambu else ''
    suffix = '+'.join(nodeid.split('::', 1)[1].replace('test_', '') for nodeid in nodeids)
    return f'{prefix}{path_to_name(path)}-{suffix}'


def make_job(name, extends, pytest_file, needs_example_model):
    return {
        f'pytest.{name}': {
            'extends': extends,
            'variables': {
                'PYTESTFILE': pytest_file,
                'EXAMPLEMODEL': int(needs_example_model),
                'VIVADO_VERSION': '2020.1',
                'VITIS_VERSION': '2024.1',
                'RUN_SYNTHESIS': 'true',
            },
        }
    }


def extra_standard_args(path):
    args = []
    if path.stem in BAMBU_SHARED_TESTS:
        args.extend(BAMBU_EXCLUDE_ARGS)
    args.extend(
        f'--ci-exclude-nodeid={nodeid}' for nodeid in sorted(BAMBU_ONLY_TESTS) if nodeid.startswith(f'{path.name}::')
    )
    return tuple(args)


def emit_or_group(path, test_root, grouped, yml, bambu=False, extra_args=()):
    extends = pytest_template(path, bambu=bambu)
    selector = str(path.relative_to(test_root))
    item = {
        'path': path,
        'name': job_name([path], bambu=bambu),
        'selector': selector,
        'needs_example_model': uses_example_model(path),
    }

    if path.stem in LONGLIST:
        pytest_file = ' '.join([selector, *extra_args]).strip()
        yml.update(make_job(item['name'], extends, pytest_file, item['needs_example_model']))
    else:
        grouped.setdefault((extends, extra_args), []).append(item)


def emit_bambu_only_tests(path, yml):
    for nodeid in sorted(BAMBU_ONLY_TESTS):
        if not nodeid.startswith(f'{path.name}::'):
            continue
        name = f'bambu-{path_to_name(path)}-{nodeid.split("::", 1)[1].replace("test_", "")}'
        yml.update(make_job(name, pytest_template(path, bambu=True), nodeid, uses_example_model(path)))


def emit_split_test_jobs(path, test_root, yml):
    functions = collect_test_functions_from_ast(path, test_root)
    chunk_size = SPLIT_BY_TEST_CASE[path.stem]
    needs_example_model = uses_example_model(path)

    for batch in batched(functions, chunk_size):
        standard_pytest_file = ' '.join([*batch, *extra_standard_args(path)]).strip()
        yml.update(make_job(test_case_job_name(path, batch), pytest_template(path), standard_pytest_file, needs_example_model))

        if path.stem in BAMBU_SHARED_TESTS:
            bambu_pytest_file = ' '.join([*batch, *BAMBU_FILTER_ARGS]).strip()
            yml.update(
                make_job(
                    test_case_job_name(path, batch, bambu=True),
                    pytest_template(path, bambu=True),
                    bambu_pytest_file,
                    needs_example_model,
                )
            )


def generate_test_yaml(test_root='.'):
    test_root = Path(test_root)
    yml = {}
    grouped = {}

    test_paths = [path for path in test_root.glob('**/test_*.py') if path.stem not in BLACKLIST]

    for path in sorted(test_paths):
        if path.stem in SPLIT_BY_TEST_CASE:
            emit_split_test_jobs(path, test_root, yml)
            continue

        emit_or_group(path, test_root, grouped, yml, extra_args=extra_standard_args(path))
        if path.stem in BAMBU_SHARED_TESTS:
            emit_or_group(path, test_root, grouped, yml, bambu=True, extra_args=BAMBU_FILTER_ARGS)
        emit_bambu_only_tests(path, yml)

    for (extends, extra_args), items in grouped.items():
        items = sorted(items, key=lambda item: f'{item["needs_example_model"]}_{item["name"]}')
        for batch in batched(items, n_test_files_per_yml):
            paths = [item['path'] for item in batch]
            selectors = [item['selector'] for item in batch]
            pytest_file = ' '.join([*selectors, *extra_args]).strip()
            needs_example_model = any(item['needs_example_model'] for item in batch)
            name = job_name(paths, bambu=extends.startswith('.pytest-bambu'))
            yml.update(make_job(name, extends, pytest_file, needs_example_model))

    return yml


if __name__ == '__main__':
    yml = generate_test_yaml(Path(__file__).parent)
    with open('pytests.yml', 'w') as yamlfile:
        yaml.safe_dump(yml, yamlfile)
