import os
import subprocess
import tempfile

from git import Repo
from supermutes.dot import dotify

DEFAULT_GIT_BRANCH = "develop"


def git_clone(repo_url, branch="master", path=""):
    """clones repo to a temporary dir, the path of which is determined by the platform"""

    _tmp_dir = tempfile.mkdtemp(prefix="mlbench-")
    repo = Repo.clone_from(repo_url, _tmp_dir, branch=branch)

    return os.path.join(_tmp_dir, path)


class ChartBuilder:
    """Class that allows building helm charts either from a repository or a local folder

    Args:
        chart (dict): Dictionary describing the location. Should look like:
            ```
            {
             "name": [chart_name],
             "source": {
                        "type": ["git" or "directory"],
                        "location": [repo_url or directory path]
                        "reference": [optional, to select the branch]
                        }
            }
            ```
    """

    def __init__(self, chart):
        self.chart = dotify(chart)
        self.source_directory = self.source_clone()

    def source_clone(self):
        """
        Clone the charts source
        We only support a git source type right now, which can also
        handle git:// local paths as well
        """

        subpath = self.chart.source.get("subpath", "")

        if "name" not in self.chart:
            raise ValueError("Please specify name for the chart")

        if "type" not in self.chart.source:
            raise ValueError("Need source type for chart {}".format(self.chart.name))

        if self.chart.source.type == "git":
            if "reference" not in self.chart.source:
                self.chart.source.reference = DEFAULT_GIT_BRANCH
            if "path" not in self.chart.source:
                self.chart.source.path = ""
            self._source_tmp_dir = git_clone(
                self.chart.source.location,
                self.chart.source.reference,
                self.chart.source.path,
            )
        elif self.chart.source.type == "directory":
            self._source_tmp_dir = self.chart.source.location

        else:
            raise ValueError(
                "Unknown source type %s for chart %s",
                self.chart.name,
                self.chart.source.type,
            )

        return os.path.join(self._source_tmp_dir, subpath)

    def _get_values_string(self, vals, parent=None):
        """Given a dictionary of values, recursively returns the arguments to pass to `helm template`.

        For example: {"key1": "value1", "key2": {"key3":"value3"}}
            gives ["--set", "key1=value1", "--set", "key2.key3=value3"]

        Args:
            vals (dict): Dictionary of values
            parent (str, optional): The parent key

        Returns:
            (list[str]): The command list
        """
        values = []
        for k, v in vals.items():
            if type(v) == dict:
                values += self._get_values_string(v, k)
            else:
                key = "{}={}".format(k, v)
                if parent is not None:
                    key = "{}.{}".format(parent, key)

                values += ["--set", key]
        return values

    def get_chart(self, release_name, values):
        """Executes the command `helm template {args}` to generate the chart
        and saves the yaml to a temporary directory

        Args:
            release_name (str): Release name
            values (dict): Values to overwrite

        Returns:
            (str): Path of generated template
        """
        values_options = self._get_values_string(values)
        output = subprocess.check_output(
            ["helm", "template", release_name, self.source_directory] + values_options
        )

        if self.chart.source.type == "git":
            subpath = self.chart.source.get("subpath", "")
            template_path = os.path.join(
                self._source_tmp_dir, subpath, "mlbench_template.yaml"
            )
        else:
            template_path = os.path.join(tempfile.mkdtemp(), "template.yaml")

        with open(template_path, "wb") as f:
            f.write(output)
        return template_path
