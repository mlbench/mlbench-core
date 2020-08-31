Developer Docs
==============


Local Dev Setup
---------------
- Clone the repo locally
- install dependencies with `pip install .[test]`
- Setup pre-commit hooks using `pre-commit install`

Tests can be run using `tox` or `pytest` commands.

Docs can be built using `make docs` command.


Making a release
----------------

Steps to make a release:

- Create a development branch based on current `develop`, named `release/vX.X.X` (e.g. `release/2.4.1`)
- Use bumpversion to bump the version, e.g. `bumpversion --verbose --no-commit --no-tag minor` to bump the minor version (`major`, `minor`, `patch` and `dev` are supported)
- Generate the new changelog (based on github issues) like `github_changelog_generator -u mlbench -p mlbench-core -t <github_token> --release-branch release/2.4.1 --future-release 2.4.1 --base CHANGELOG.md` (use a valid `<github_token>`)
  found here https://github.com/github-changelog-generator/github-changelog-generator
  Convert the resulting Changelog.md file to *.rst with a tool like https://cloudconvert.com/md-to-rst . Use this to update the `changelog.rst` in the `mlbench-docs` repo.
- Commit the changes and merge the `release/X.X.X` branch into both master and develop and push with `git push`.
- Create a tag of the master version using `git tag -m "Release X.X.X" vX.X.X` and push with `git push --all`
- Build with `python setup.py sdist bdist_wheel` (delete `dist/` before building) and the upload to Pypi with `twine upload dist/*`