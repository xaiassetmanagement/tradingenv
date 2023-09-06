"""https://nbconvert.readthedocs.io/en/latest/execute_api.html"""
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
import pytest
from pathlib import Path
here = Path(os.path.abspath(os.path.dirname(__file__)))


def ipynb_paths():
    paths = list()
    for root, dirs, files in os.walk(here):
        folder = os.path.basename(root)
        if not folder.startswith('.'):
            for file in files:
                if file.endswith(".ipynb") and not file.startswith('.'):
                    paths.append(os.path.join(root, file))
    return paths


class TestExamples:
    @pytest.mark.parametrize('path_to_ipynb', ipynb_paths())
    def test_notebook(self, path_to_ipynb):
        """goal is to load all .ipynb from /notebooks (v), run them (v)
        and then save them as html with toc.
        tutorial: https://nbconvert.readthedocs.io/en/latest/nbconvert_library.html

        Notes
        -----
        Rendering cufflinks charts in jupyter lab requires the extension
        described at the following link:
        https://medium.com/@hicraigchen/plotly-with-pandas-via-cufflinks-in-jupyter-lab-issues-50fcf1a89a1c
        """
        with open(path_to_ipynb) as f:
            nb = nbformat.read(f, as_version=4)

        # Configure the execution mode.
        ep = ExecutePreprocessor(timeout=-1, kernel_name='python3')

        # Execute/Run (preprocess):
        ep.preprocess(nb)

        # 2. Instantiate the exporter. We use the `basic` template for now;
        # we'll get into more details later about how to customize the
        # exporter further.
        html_exporter = HTMLExporter()
        html_as_string, resources = html_exporter.from_notebook_node(nb)

        # Save.
        dir_examples_html = os.path.join(here, 'html')
        os.makedirs(dir_examples_html, exist_ok=True)
        html_name = os.path.basename(path_to_ipynb).replace('ipynb', 'html')
        html_path = os.path.join(dir_examples_html, html_name)

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_as_string)
