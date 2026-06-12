import ast
import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, Iterable, List, Optional, Sequence, Tuple

import yaml

"""
Create a Gitlab CI yml file with separate pytest jobs.

Bambu tests run in a dedicated image. The generator does not import test
modules; it only parses parametrization decorators to decide whether a file or
function contains Bambu backend cases.
"""

n_test_files_per_yml = int(os.environ.get('N_TESTS_PER_YAML', 4))

BAMBU_BACKENDS = {'Bambu', 'BambuAccelerator'}
BAMBU_FILTER_ARGS = tuple(f'--backend-filter={backend}' for backend in sorted(BAMBU_BACKENDS))
BAMBU_EXCLUDE_ARGS = tuple(f'--backend-exclude={backend}' for backend in sorted(BAMBU_BACKENDS))

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

# Test files to split by individual test functions
# Value = chunk size per CI job
SPLIT_BY_TEST_CASE = {
    'test_keras_api': 1,
}

# Bambu-specific tests that are not expressed by backend parametrization
BAMBU_ONLY_TESTS = {
    'test_report.py::test_bambu_report',
}

# Keep Bambu synthesis/build tests isolated from unrelated Bambu jobs
BAMBU_STANDALONE = {'test_build_bambu'}


@dataclass(frozen=True)
class TestFunction:
    path: Path
    name: str
    # None means the backend parametrization exists but was not statically evaluable.
    backend_values: Optional[FrozenSet[str]]

    def nodeid(self, test_root: Path) -> str:
        return f'{self.path.relative_to(test_root)}::{self.name}'


@dataclass
class Selection:
    name: str
    selectors: List[str]
    extends: str
    needs_example_model: bool
    extra_args: Tuple[str, ...] = field(default_factory=tuple)
    standalone: bool = False

    @property
    def pytest_file_arg(self):
        return ' '.join([*self.selectors, *self.extra_args]).strip()


def batched(iterable: Iterable, batch_size: int):
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, batch_size)):
        yield batch


def path_to_name(test_path):
    return Path(test_path).stem.replace('test_', '')


def uses_example_model(test_filename):
    with open(test_filename) as f:
        return 'example-models' in f.read()


def _literal_eval(node, constants):
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.List):
        return [_literal_eval(elt, constants) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal_eval(elt, constants) for elt in node.elts)
    return ast.literal_eval(node)


def _module_constants(tree):
    constants = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            constants[target.id] = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
    return constants


def _parametrize_call(decorator):
    if not isinstance(decorator, ast.Call):
        return False
    func = decorator.func
    return isinstance(func, ast.Attribute) and func.attr == 'parametrize' and len(decorator.args) >= 2


def _parametrize_argnames(call, constants):
    argnames = _literal_eval(call.args[0], constants)
    if isinstance(argnames, str):
        return [arg.strip() for arg in argnames.split(',')]
    return list(argnames)


def _parametrize_backend_values(function_node, constants):
    backend_values = set()
    saw_backend_parametrize = False

    for decorator in function_node.decorator_list:
        if not _parametrize_call(decorator):
            continue

        try:
            argnames = _parametrize_argnames(decorator, constants)
        except (ValueError, SyntaxError):
            continue

        if 'backend' not in argnames:
            continue

        saw_backend_parametrize = True
        backend_index = argnames.index('backend')

        try:
            argvalues = _literal_eval(decorator.args[1], constants)
        except (ValueError, SyntaxError):
            return None

        for value in argvalues:
            values = (value,) if len(argnames) == 1 else tuple(value)
            backend = values[backend_index]
            if isinstance(backend, str):
                backend_values.add(backend)

    if not saw_backend_parametrize:
        return frozenset()
    if not backend_values:
        return None
    return frozenset(backend_values)


def collect_test_functions_from_ast(test_file, test_root):
    """Collect test function metadata using AST parsing, without importing tests."""
    with open(test_file, encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=str(test_file))

    constants = _module_constants(tree)
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test'):
            functions.append(
                TestFunction(
                    path=Path(test_file),
                    name=node.name,
                    backend_values=_parametrize_backend_values(node, constants),
                )
            )
    return functions


def _profile_for_path(path, bambu=False):
    keras3 = path.stem in KERAS3_LIST
    if bambu and keras3:
        return '.pytest-bambu-keras3-only'
    if bambu:
        return '.pytest-bambu'
    if keras3:
        return '.pytest-keras3-only'
    return '.pytest'


def _selection_name(path, selectors, bambu=False):
    prefix = 'bambu-' if bambu else ''
    if len(selectors) == 1 and '::' in selectors[0]:
        return f'{prefix}{path_to_name(path)}-{selectors[0].split("::", 1)[1].replace("test_", "")}'
    return f'{prefix}{"+".join(path_to_name(selector) for selector in selectors)}'


def _make_selection(path, test_root, bambu=False, selectors=None, extra_args=(), standalone=False):
    selectors = selectors or [str(path.relative_to(test_root))]
    return Selection(
        name=_selection_name(path, selectors, bambu=bambu),
        selectors=selectors,
        extends=_profile_for_path(path, bambu=bambu),
        needs_example_model=uses_example_model(path),
        extra_args=tuple(extra_args),
        standalone=standalone,
    )


def _has_bambu_cases(functions):
    return any(
        function.backend_values is not None and bool(function.backend_values & BAMBU_BACKENDS) for function in functions
    )


def _standard_needed(function, test_root):
    if function.nodeid(test_root) in BAMBU_ONLY_TESTS:
        return False
    if function.backend_values is None:
        return True
    if not function.backend_values:
        return True
    return bool(function.backend_values - BAMBU_BACKENDS)


def _has_standard_cases(functions, test_root):
    return any(_standard_needed(function, test_root) for function in functions)


def _bambu_only_nodeids(functions, test_root):
    return [function.nodeid(test_root) for function in functions if function.nodeid(test_root) in BAMBU_ONLY_TESTS]


def _file_selections(path, test_root, functions):
    selections = []
    bambu_only = _bambu_only_nodeids(functions, test_root)
    standard_args = tuple(f'--ci-exclude-nodeid={nodeid}' for nodeid in bambu_only)
    if _has_bambu_cases(functions):
        standard_args = BAMBU_EXCLUDE_ARGS + standard_args

    if _has_standard_cases(functions, test_root):
        selections.append(
            _make_selection(
                path,
                test_root,
                extra_args=standard_args,
                standalone=path.stem in LONGLIST,
            )
        )

    if _has_bambu_cases(functions):
        selections.append(
            _make_selection(
                path,
                test_root,
                bambu=True,
                extra_args=BAMBU_FILTER_ARGS,
                standalone=path.stem in LONGLIST or path.stem in BAMBU_STANDALONE,
            )
        )

    for nodeid in bambu_only:
        selections.append(_make_selection(path, test_root, bambu=True, selectors=[nodeid], standalone=True))

    return selections


def _split_function_selections(path, test_root, functions):
    selections = []
    chunk_size = SPLIT_BY_TEST_CASE[path.stem]

    for batch in batched(functions, chunk_size):
        standard_selectors = [function.nodeid(test_root) for function in batch if _standard_needed(function, test_root)]
        bambu_selectors = [
            function.nodeid(test_root)
            for function in batch
            if function.backend_values is not None and bool(function.backend_values & BAMBU_BACKENDS)
        ]

        if standard_selectors:
            needs_exclude = any(
                function.backend_values is not None and bool(function.backend_values & BAMBU_BACKENDS)
                for function in batch
            )
            selections.append(
                _make_selection(
                    path,
                    test_root,
                    selectors=standard_selectors,
                    extra_args=BAMBU_EXCLUDE_ARGS if needs_exclude else (),
                    standalone=True,
                )
            )

        if bambu_selectors:
            selections.append(
                _make_selection(
                    path,
                    test_root,
                    bambu=True,
                    selectors=bambu_selectors,
                    extra_args=BAMBU_FILTER_ARGS,
                    standalone=True,
                )
            )

    return selections


def collect_selections(test_root):
    selections = []
    paths = [path for path in test_root.glob('**/test_*.py') if path.stem not in BLACKLIST]

    for path in sorted(paths):
        functions = collect_test_functions_from_ast(path, test_root)
        if path.stem in SPLIT_BY_TEST_CASE:
            selections.extend(_split_function_selections(path, test_root, functions))
        else:
            selections.extend(_file_selections(path, test_root, functions))

    return selections


def _job_dict(selection):
    return {
        f'pytest.{selection.name}': {
            'extends': selection.extends,
            'variables': {
                'PYTESTFILE': selection.pytest_file_arg,
                'EXAMPLEMODEL': int(selection.needs_example_model),
                'VIVADO_VERSION': '2020.1',
                'VITIS_VERSION': '2024.1',
                'RUN_SYNTHESIS': 'true',
            },
        }
    }


def _merge_batch(batch: Sequence[Selection]):
    selectors = [selector for selection in batch for selector in selection.selectors]
    extra_args = []
    seen_args = set()
    for selection in batch:
        for arg in selection.extra_args:
            if arg not in seen_args:
                extra_args.append(arg)
                seen_args.add(arg)

    return Selection(
        name='+'.join(selection.name for selection in batch),
        selectors=selectors,
        extends=batch[0].extends,
        needs_example_model=any(selection.needs_example_model for selection in batch),
        extra_args=tuple(extra_args),
    )


def generate_test_yaml(test_root='.'):
    test_root = Path(test_root)
    selections = collect_selections(test_root)
    yml = {}

    grouped = {}
    for selection in selections:
        if selection.standalone:
            yml.update(_job_dict(selection))
        else:
            grouped.setdefault(selection.extends, []).append(selection)

    for batchable_selections in grouped.values():
        batchable_selections = sorted(
            batchable_selections,
            key=lambda selection: f'{selection.needs_example_model}_{selection.name}',
        )
        for batch in batched(batchable_selections, n_test_files_per_yml):
            yml.update(_job_dict(_merge_batch(batch)))

    return yml


if __name__ == '__main__':
    yml = generate_test_yaml(Path(__file__).parent)
    with open('pytests.yml', 'w') as yamlfile:
        yaml.safe_dump(yml, yamlfile)
