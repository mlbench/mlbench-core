# Changelog

## [v3.0.0](https://github.com/mlbench/mlbench-core/tree/v3.0.0) (2020-12-07)

[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v2.4.0...v3.0.0)

**Implemented enhancements:**

- Support multiple clusters in CLI [\#91](https://github.com/mlbench/mlbench-core/issues/91)
- Add notebook/code to visualize results [\#72](https://github.com/mlbench/mlbench-core/issues/72)
- Support AWS in CLI [\#33](https://github.com/mlbench/mlbench-core/issues/33)
- Fix rnn language model [\#303](https://github.com/mlbench/mlbench-core/pull/303) ([ehoelzl](https://github.com/ehoelzl))
- Transformer language translation [\#99](https://github.com/mlbench/mlbench-core/pull/99) ([ehoelzl](https://github.com/ehoelzl))

**Fixed bugs:**

- Training code keeps running for PyTorch after training is done [\#26](https://github.com/mlbench/mlbench-core/issues/26)

**Closed issues:**

- Remove loss argument for metric computation [\#295](https://github.com/mlbench/mlbench-core/issues/295)
- Update PyTorch to 1.7 [\#286](https://github.com/mlbench/mlbench-core/issues/286)
- Refactor optimizer and chose more appropriate names [\#284](https://github.com/mlbench/mlbench-core/issues/284)
- fails to create kind cluster [\#277](https://github.com/mlbench/mlbench-core/issues/277)
- Refactor CLI [\#253](https://github.com/mlbench/mlbench-core/issues/253)
- Dependabot couldn't authenticate with https://pypi.python.org/simple/ [\#252](https://github.com/mlbench/mlbench-core/issues/252)
- Unify requirements/setup.py versions [\#244](https://github.com/mlbench/mlbench-core/issues/244)
- isort failing on all PRs [\#227](https://github.com/mlbench/mlbench-core/issues/227)
- torch.div is not supported in PyTorch 1.6 [\#223](https://github.com/mlbench/mlbench-core/issues/223)
- Refactor common functionality for tiller and helm [\#108](https://github.com/mlbench/mlbench-core/issues/108)
- Add GPU support for AWS in CLI [\#104](https://github.com/mlbench/mlbench-core/issues/104)
- Change CPU limit to \#CPUs - 1 [\#101](https://github.com/mlbench/mlbench-core/issues/101)
- Add --version flag [\#97](https://github.com/mlbench/mlbench-core/issues/97)
- Cluster creation/deletion errors with non-default zone [\#94](https://github.com/mlbench/mlbench-core/issues/94)
- Add command to list runs [\#86](https://github.com/mlbench/mlbench-core/issues/86)
- RefreshError from gcloud [\#83](https://github.com/mlbench/mlbench-core/issues/83)
- Run new benchmarks and document costs [\#82](https://github.com/mlbench/mlbench-core/issues/82)
- Make nvidia k80 default GPU [\#80](https://github.com/mlbench/mlbench-core/issues/80)
- Fix random seeds [\#79](https://github.com/mlbench/mlbench-core/issues/79)
- benchmark against torch.nn.parallel.DistributedDataParallel MPSG [\#75](https://github.com/mlbench/mlbench-core/issues/75)
- upgrade to pytorch 1.5 [\#74](https://github.com/mlbench/mlbench-core/issues/74)
- Provide comparison to competitors [\#66](https://github.com/mlbench/mlbench-core/issues/66)
- Add some integration tests [\#64](https://github.com/mlbench/mlbench-core/issues/64)
- Remove stale branches [\#62](https://github.com/mlbench/mlbench-core/issues/62)
- Add PowerSGD optimizer [\#59](https://github.com/mlbench/mlbench-core/issues/59)
- Add RNN Language Model [\#54](https://github.com/mlbench/mlbench-core/issues/54)
- Use torch.nn.DataParallel for intra-node computation [\#46](https://github.com/mlbench/mlbench-core/issues/46)
- Add CLI support for DIND [\#42](https://github.com/mlbench/mlbench-core/issues/42)
- Port over functionality from Language Model benchmark to the core library [\#34](https://github.com/mlbench/mlbench-core/issues/34)
- make results reproducible from command-line [\#24](https://github.com/mlbench/mlbench-core/issues/24)
- Contribution and docs section on README.md [\#17](https://github.com/mlbench/mlbench-core/issues/17)
- test new  torch.distributed  [\#15](https://github.com/mlbench/mlbench-core/issues/15)

**Merged pull requests:**

- Bugfix KIND cli [\#307](https://github.com/mlbench/mlbench-core/pull/307) ([ehoelzl](https://github.com/ehoelzl))
- Update README.md to show new badge [\#306](https://github.com/mlbench/mlbench-core/pull/306) ([ehoelzl](https://github.com/ehoelzl))
- Create manual.yml [\#305](https://github.com/mlbench/mlbench-core/pull/305) ([ehoelzl](https://github.com/ehoelzl))
- Switch to github actions [\#304](https://github.com/mlbench/mlbench-core/pull/304) ([ehoelzl](https://github.com/ehoelzl))
- Bump sphinx from 3.3.0 to 3.3.1 [\#301](https://github.com/mlbench/mlbench-core/pull/301) ([dependabot[bot]](https://github.com/apps/dependabot))
- Remove loss from metric argument [\#297](https://github.com/mlbench/mlbench-core/pull/297) ([ehoelzl](https://github.com/ehoelzl))
- Fix translators [\#294](https://github.com/mlbench/mlbench-core/pull/294) ([ehoelzl](https://github.com/ehoelzl))
- Update pytorch [\#292](https://github.com/mlbench/mlbench-core/pull/292) ([ehoelzl](https://github.com/ehoelzl))
- Bump sphinx from 3.2.1 to 3.3.0 in /docs [\#288](https://github.com/mlbench/mlbench-core/pull/288) ([dependabot[bot]](https://github.com/apps/dependabot))
- Refactor optimizers [\#285](https://github.com/mlbench/mlbench-core/pull/285) ([ehoelzl](https://github.com/ehoelzl))
- Bump isort from 5.5.4 to 5.6.4 [\#283](https://github.com/mlbench/mlbench-core/pull/283) ([dependabot[bot]](https://github.com/apps/dependabot))
- Bump sphinx-autoapi from 1.5.0 to 1.5.1 [\#280](https://github.com/mlbench/mlbench-core/pull/280) ([dependabot[bot]](https://github.com/apps/dependabot))
- Add gpu functionality on AWS [\#278](https://github.com/mlbench/mlbench-core/pull/278) ([mmilenkoski](https://github.com/mmilenkoski))
- Catch exceptions when creating/deleting clusters [\#276](https://github.com/mlbench/mlbench-core/pull/276) ([ehoelzl](https://github.com/ehoelzl))
- Fix doc [\#275](https://github.com/mlbench/mlbench-core/pull/275) ([ehoelzl](https://github.com/ehoelzl))
- Fix AWS deployment [\#274](https://github.com/mlbench/mlbench-core/pull/274) ([mmilenkoski](https://github.com/mmilenkoski))
- Create dependabot.yml [\#260](https://github.com/mlbench/mlbench-core/pull/260) ([ehoelzl](https://github.com/ehoelzl))
- Merge requirements & Update doc [\#259](https://github.com/mlbench/mlbench-core/pull/259) ([ehoelzl](https://github.com/ehoelzl))
- Bump google-api-python-client from 1.9.3 to 1.12.1 [\#246](https://github.com/mlbench/mlbench-core/pull/246) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump numpy from 1.19.0 to 1.19.2 [\#245](https://github.com/mlbench/mlbench-core/pull/245) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump boto3 from 1.14.6 to 1.14.50 [\#234](https://github.com/mlbench/mlbench-core/pull/234) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Fix isort errors [\#233](https://github.com/mlbench/mlbench-core/pull/233) ([mmilenkoski](https://github.com/mmilenkoski))
- Bump pytest-mock from 3.1.1 to 3.3.1 [\#231](https://github.com/mlbench/mlbench-core/pull/231) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump isort from 4.3.21 to 5.4.2 [\#221](https://github.com/mlbench/mlbench-core/pull/221) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump sphinx from 3.0.4 to 3.2.1 [\#220](https://github.com/mlbench/mlbench-core/pull/220) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump grpcio from 1.29.0 to 1.31.0 [\#207](https://github.com/mlbench/mlbench-core/pull/207) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump spacy from 2.3.0 to 2.3.2 [\#182](https://github.com/mlbench/mlbench-core/pull/182) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Downgrade Sphinx [\#162](https://github.com/mlbench/mlbench-core/pull/162) ([ehoelzl](https://github.com/ehoelzl))
- Add developer docs [\#161](https://github.com/mlbench/mlbench-core/pull/161) ([Panaetius](https://github.com/Panaetius))
- Fp optimizer changes [\#160](https://github.com/mlbench/mlbench-core/pull/160) ([ehoelzl](https://github.com/ehoelzl))
- Bump wcwidth from 0.1.9 to 0.2.5 [\#156](https://github.com/mlbench/mlbench-core/pull/156) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump all versions and add doc test [\#152](https://github.com/mlbench/mlbench-core/pull/152) ([Panaetius](https://github.com/Panaetius))
- Bump torchvision from 0.6.0 to 0.6.1 [\#151](https://github.com/mlbench/mlbench-core/pull/151) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump numpy from 1.18.5 to 1.19.0 [\#150](https://github.com/mlbench/mlbench-core/pull/150) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump torch from 1.5.0 to 1.5.1 [\#148](https://github.com/mlbench/mlbench-core/pull/148) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump google-auth from 1.17.2 to 1.18.0 [\#147](https://github.com/mlbench/mlbench-core/pull/147) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump sphinx-rtd-theme from 0.4.3 to 0.5.0 [\#144](https://github.com/mlbench/mlbench-core/pull/144) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump spacy from 2.2.4 to 2.3.0 [\#142](https://github.com/mlbench/mlbench-core/pull/142) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump sphinx from 3.1.0 to 3.1.1 [\#140](https://github.com/mlbench/mlbench-core/pull/140) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump dill from 0.3.1.1 to 0.3.2 [\#138](https://github.com/mlbench/mlbench-core/pull/138) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Update dependencies [\#137](https://github.com/mlbench/mlbench-core/pull/137) ([Panaetius](https://github.com/Panaetius))
- Bump spacy from 2.2.3 to 2.2.4 [\#135](https://github.com/mlbench/mlbench-core/pull/135) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump numpy from 1.16.6 to 1.18.5 [\#133](https://github.com/mlbench/mlbench-core/pull/133) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump freezegun from 0.3.12 to 0.3.15 [\#129](https://github.com/mlbench/mlbench-core/pull/129) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump tabulate from 0.8.6 to 0.8.7 [\#128](https://github.com/mlbench/mlbench-core/pull/128) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump deprecation from 2.0.6 to 2.1.0 [\#125](https://github.com/mlbench/mlbench-core/pull/125) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump pytest-black from 0.3.8 to 0.3.9 [\#124](https://github.com/mlbench/mlbench-core/pull/124) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump sphinx-rtd-theme from 0.4.2 to 0.4.3 [\#123](https://github.com/mlbench/mlbench-core/pull/123) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump sphinx from 1.8.1 to 3.1.0 [\#121](https://github.com/mlbench/mlbench-core/pull/121) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump pytest-mock from 1.10.0 to 3.1.1 [\#120](https://github.com/mlbench/mlbench-core/pull/120) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump torchtext from 0.5.0 to 0.6.0 [\#118](https://github.com/mlbench/mlbench-core/pull/118) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump torchvision from 0.5.0 to 0.6.0 [\#117](https://github.com/mlbench/mlbench-core/pull/117) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Adds support for multiple clusters [\#115](https://github.com/mlbench/mlbench-core/pull/115) ([Panaetius](https://github.com/Panaetius))
- Bump click from 7.0 to 7.1.2 [\#114](https://github.com/mlbench/mlbench-core/pull/114) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump google-cloud-container from 0.3.0 to 0.5.0 [\#113](https://github.com/mlbench/mlbench-core/pull/113) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump appdirs from 1.4.3 to 1.4.4 [\#112](https://github.com/mlbench/mlbench-core/pull/112) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump sphinxcontrib-bibtex from 0.4.0 to 1.0.0 [\#111](https://github.com/mlbench/mlbench-core/pull/111) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Bump sphinx-autoapi from 1.3.0 to 1.4.0 [\#110](https://github.com/mlbench/mlbench-core/pull/110) ([dependabot-preview[bot]](https://github.com/apps/dependabot-preview))
- Remove unused arguments in create\_aws [\#109](https://github.com/mlbench/mlbench-core/pull/109) ([mmilenkoski](https://github.com/mmilenkoski))
- Fix Random seeds, Add new tracker stats [\#107](https://github.com/mlbench/mlbench-core/pull/107) ([ehoelzl](https://github.com/ehoelzl))
- Add return\_code check in test\_cli [\#106](https://github.com/mlbench/mlbench-core/pull/106) ([mmilenkoski](https://github.com/mmilenkoski))
- Add AWS support in CLI [\#103](https://github.com/mlbench/mlbench-core/pull/103) ([mmilenkoski](https://github.com/mmilenkoski))
- Update test\_cli.py [\#100](https://github.com/mlbench/mlbench-core/pull/100) ([giorgiosav](https://github.com/giorgiosav))
- Adds a chart command to cli [\#95](https://github.com/mlbench/mlbench-core/pull/95) ([Panaetius](https://github.com/Panaetius))
- Add support for kind cluster creation in the CLI [\#93](https://github.com/mlbench/mlbench-core/pull/93) ([mmilenkoski](https://github.com/mmilenkoski))

# Changelog

## [v2.4.0](https://github.com/mlbench/mlbench-core/tree/v2.4.0) (2020-04-20)

[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v2.3.2...v2.4.0)

**Implemented enhancements:**

- Switch to black for code formatting [\#35](https://github.com/mlbench/mlbench-core/issues/35)

**Closed issues:**

- Travis tests run only for Python 3.6 [\#65](https://github.com/mlbench/mlbench-core/issues/65)
- Downloading results fails if `--output` option is not provided [\#57](https://github.com/mlbench/mlbench-core/issues/57)
- Remember user input in mlbench run [\#56](https://github.com/mlbench/mlbench-core/issues/56)
- Aggregate the gradients by model, instead of by layers. [\#45](https://github.com/mlbench/mlbench-core/issues/45)
- Update docker images to CUDA10, mlbench-core module to newest [\#43](https://github.com/mlbench/mlbench-core/issues/43)
- Upgrade PyTorch to 1.4 [\#40](https://github.com/mlbench/mlbench-core/issues/40)

**Merged pull requests:**

- Pytorch v1.4.0 [\#68](https://github.com/mlbench/mlbench-core/pull/68) ([ehoelzl](https://github.com/ehoelzl))
- Fix ci [\#67](https://github.com/mlbench/mlbench-core/pull/67) ([ehoelzl](https://github.com/ehoelzl))
- Add aggregation by model [\#61](https://github.com/mlbench/mlbench-core/pull/61) ([ehoelzl](https://github.com/ehoelzl))
- Remember user input in mlbench run [\#60](https://github.com/mlbench/mlbench-core/pull/60) ([mmilenkoski](https://github.com/mmilenkoski))
- Add default name of output file in CLI [\#58](https://github.com/mlbench/mlbench-core/pull/58) ([mmilenkoski](https://github.com/mmilenkoski))
- Cli adaptation [\#55](https://github.com/mlbench/mlbench-core/pull/55) ([ehoelzl](https://github.com/ehoelzl))
- Update tags and patch version to 2.3.2 [\#52](https://github.com/mlbench/mlbench-core/pull/52) ([ehoelzl](https://github.com/ehoelzl))
- Add get\_optimizer to create optimizer object [\#48](https://github.com/mlbench/mlbench-core/pull/48) ([mmilenkoski](https://github.com/mmilenkoski))

# Changelog

## [v2.3.2](https://github.com/mlbench/mlbench-core/tree/v2.3.2) (2020-04-07)

[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v2.3.1...v2.3.2)

**Implemented enhancements:**

- Add NCCL & GLOO Backend support [\#49](https://github.com/mlbench/mlbench-core/issues/49)
- Add NCCL & GLOO Backend support [\#47](https://github.com/mlbench/mlbench-core/pull/47) ([giorgiosav](https://github.com/giorgiosav))

**Fixed bugs:**

- math ValueError with 1-node cluster [\#38](https://github.com/mlbench/mlbench-core/issues/38)

**Merged pull requests:**

- num\_workers fix [\#51](https://github.com/mlbench/mlbench-core/pull/51) ([giorgiosav](https://github.com/giorgiosav))
- Adds centralized Adam implementation [\#41](https://github.com/mlbench/mlbench-core/pull/41) ([mmilenkoski](https://github.com/mmilenkoski))

# Change Log

## [2.3.1](https://github.com/mlbench/mlbench-core/tree/2.3.1) (2020-03-09)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v2.3.0...2.3.1)

**Implemented enhancements:**

- Customize Communication Scheme For Sparsified/Quantizatized/Decentralized scenarios [\#12](https://github.com/mlbench/mlbench-core/issues/12)

## [v2.3.0](https://github.com/mlbench/mlbench-core/tree/v2.3.0) (2019-12-23)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v2.2.1...v2.3.0)

## [v2.2.1](https://github.com/mlbench/mlbench-core/tree/v2.2.1) (2019-12-16)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v2.2.0...v2.2.1)

# Change Log

## [v2.2.0](https://github.com/mlbench/mlbench-core/tree/v2.2.0) (2019-11-11)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v2.1.0...v2.1.1)

**Implemented enhancements:**
- `initialize_backends` can now be called as context manager
- Improved CLI to run multiple runs in parallel

## [v2.1.1](https://github.com/mlbench/mlbench-core/tree/v2.1.1) (2019-11-11)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v2.1.0...v2.1.1)


## [v2.1.0](https://github.com/mlbench/mlbench-core/tree/v2.1.0) (2019-11-4)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v2.0.0...v2.1.0)

**Implemented enhancements:**

- Added CLI for MLBench runs

## [v1.4.4](https://github.com/mlbench/mlbench-core/tree/v1.4.4) (2019-05-28)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.4.3...v1.4.4)


## [v1.4.3](https://github.com/mlbench/mlbench-core/tree/v1.4.3) (2019-05-23)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.4.2...v1.4.3)


## [v1.4.2](https://github.com/mlbench/mlbench-core/tree/v1.4.2) (2019-05-21)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.4.1...v1.4.2)

## [v1.4.1](https://github.com/mlbench/mlbench-core/tree/v1.4.1) (2019-05-16)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.4.0...v1.4.1)

## [v1.4.0](https://github.com/mlbench/mlbench-core/tree/v1.4.0) (2019-05-02)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.3.4...v1.4.0)

**Implemented enhancements:**

- Split Train and Validation in Tensorflow [\#22](https://github.com/mlbench/mlbench-core/issues/22)

## [v1.3.4](https://github.com/mlbench/mlbench-core/tree/v1.3.4) (2019-03-20)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.3.3...v1.3.4)

**Implemented enhancements:**

- in controlflow, don't mix train and validation [\#20](https://github.com/mlbench/mlbench-core/issues/20)

**Fixed bugs:**

- Add metrics logging for Tensorflow [\#19](https://github.com/mlbench/mlbench-core/issues/19)

## [v1.3.3](https://github.com/mlbench/mlbench-core/tree/v1.3.3) (2019-02-26)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.3.2...v1.3.3)

## [v1.3.2](https://github.com/mlbench/mlbench-core/tree/v1.3.2) (2019-02-13)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.3.1...v1.3.2)

## [v1.3.1](https://github.com/mlbench/mlbench-core/tree/v1.3.1) (2019-02-13)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.3.0...v1.3.1)

## [v1.3.0](https://github.com/mlbench/mlbench-core/tree/v1.3.0) (2019-02-12)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.2.1...v1.3.0)

## [v1.2.1](https://github.com/mlbench/mlbench-core/tree/v1.2.1) (2019-01-31)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.2.0...v1.2.1)

## [v1.2.0](https://github.com/mlbench/mlbench-core/tree/v1.2.0) (2019-01-30)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.1.1...v1.2.0)

## [v1.1.1](https://github.com/mlbench/mlbench-core/tree/v1.1.1) (2019-01-09)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.1.0...v1.1.1)

## [v1.1.0](https://github.com/mlbench/mlbench-core/tree/v1.1.0) (2018-12-06)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.0.0...v1.1.0)

**Fixed bugs:**

- Bug when saving checkpoints [\#13](https://github.com/mlbench/mlbench-core/issues/13)

## [v1.0.0](https://github.com/mlbench/mlbench-core/tree/v1.0.0) (2018-11-20)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/1.4.2...v1.0.0)

**Implemented enhancements:**

- Add API Client to mlbench-core [\#6](https://github.com/mlbench/mlbench-core/issues/6)
- Move to google-style docs [\#4](https://github.com/mlbench/mlbench-core/issues/4)
- Add Imagenet Dataset for pytorch [\#3](https://github.com/mlbench/mlbench-core/issues/3)
- Move worker code to mlbench-core repo [\#1](https://github.com/mlbench/mlbench-core/issues/1)

# Change Log

## [1.4.2](https://github.com/mlbench/mlbench-core/tree/1.4.2) (2019-05-21)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.0.0...1.4.2)

**Implemented enhancements:**

- Split Train and Validation in Tensorflow [\#22](https://github.com/mlbench/mlbench-core/issues/22)
- in controlflow, don't mix train and validation [\#20](https://github.com/mlbench/mlbench-core/issues/20)

**Fixed bugs:**

- Add metrics logging for Tensorflow [\#19](https://github.com/mlbench/mlbench-core/issues/19)
- Bug when saving checkpoints [\#13](https://github.com/mlbench/mlbench-core/issues/13)

# Change Log

## [v1.4.1](https://github.com/mlbench/mlbench-core/tree/v1.4.1) (2019-05-16)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.4.0...v1.4.1)

## [1.4.0](https://github.com/mlbench/mlbench-core/tree/1.4.0) (2019-05-02)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.0.0...1.4.0)

**Implemented enhancements:**

- Split Train and Validation in Tensorflow [\#22](https://github.com/mlbench/mlbench-core/issues/22)
- in controlflow, don't mix train and validation [\#20](https://github.com/mlbench/mlbench-core/issues/20)

**Fixed bugs:**

- Add metrics logging for Tensorflow [\#19](https://github.com/mlbench/mlbench-core/issues/19)
- Bug when saving checkpoints [\#13](https://github.com/mlbench/mlbench-core/issues/13)

# Change Log

## [v1.3.4](https://github.com/mlbench/mlbench-core/tree/v1.3.4) (2019-03-20)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.3.3...v1.3.4)

**Implemented enhancements:**

- in controlflow, don't mix train and validation [\#20](https://github.com/mlbench/mlbench-core/issues/20)

**Fixed bugs:**

- Add metrics logging for Tensorflow [\#19](https://github.com/mlbench/mlbench-core/issues/19)

## [v1.3.3](https://github.com/mlbench/mlbench-core/tree/v1.3.3) (2019-02-26)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.3.2...v1.3.3)

## [v1.3.2](https://github.com/mlbench/mlbench-core/tree/v1.3.2) (2019-02-13)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.3.1...v1.3.2)

## [v1.3.1](https://github.com/mlbench/mlbench-core/tree/v1.3.1) (2019-02-13)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.3.0...v1.3.1)

## [v1.3.0](https://github.com/mlbench/mlbench-core/tree/v1.3.0) (2019-02-12)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.2.1...v1.3.0)

## [v1.2.1](https://github.com/mlbench/mlbench-core/tree/v1.2.1) (2019-01-31)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.2.0...v1.2.1)

## [v1.2.0](https://github.com/mlbench/mlbench-core/tree/v1.2.0) (2019-01-30)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.1.1...v1.2.0)

## [v1.1.1](https://github.com/mlbench/mlbench-core/tree/v1.1.1) (2019-01-09)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.1.0...v1.1.1)

# Change Log

## [v1.1.0](https://github.com/mlbench/mlbench-core/tree/v1.1.0) (2018-12-06)
[Full Changelog](https://github.com/mlbench/mlbench-core/compare/v1.0.0...v1.1.0)

**Fixed bugs:**

- Bug when saving checkpoints [\#13](https://github.com/mlbench/mlbench-core/issues/13)
- Adds Tensorflow Controlflow, Dataset and Model code
- Adds Pytorch linear models
- Adds sparsified and decentralized optimizers

## [v1.0.0](https://github.com/mlbench/mlbench-core/tree/v1.0.0) (2018-11-15)

**Implemented enhancements:**

- Add API Client to mlbench-core [\#6](https://github.com/mlbench/mlbench-core/issues/6)
- Move to google-style docs [\#4](https://github.com/mlbench/mlbench-core/issues/4)
- Add Imagenet Dataset for pytorch [\#3](https://github.com/mlbench/mlbench-core/issues/3)
- Move worker code to mlbench-core repo [\#1](https://github.com/mlbench/mlbench-core/issues/1)

## [0.1.0](https://github.com/mlbench/mlbench/tree/0.1.0) (2018-09-14)
**Implemented enhancements:**

- Add documentation in reference implementation to docs [\#46](https://github.com/mlbench/mlbench/issues/46)
- Replace cAdvisor with Kubernetes stats for Resource usage [\#38](https://github.com/mlbench/mlbench/issues/38)
- Rename folders [\#31](https://github.com/mlbench/mlbench/issues/31)
- Change docker image names [\#30](https://github.com/mlbench/mlbench/issues/30)
- Add continuous output for mpirun [\#27](https://github.com/mlbench/mlbench/issues/27)
- Replace SQlite with Postgres [\#25](https://github.com/mlbench/mlbench/issues/25)
- Fix unittest [\#23](https://github.com/mlbench/mlbench/issues/23)
- Add/Fix CI/Automated build [\#22](https://github.com/mlbench/mlbench/issues/22)
- Cleanup unneeded project files [\#21](https://github.com/mlbench/mlbench/issues/21)
- Remove hardcoded values [\#20](https://github.com/mlbench/mlbench/issues/20)
- Improves Notes.txt [\#19](https://github.com/mlbench/mlbench/issues/19)
- Rename components [\#15](https://github.com/mlbench/mlbench/issues/15)

**Fixed bugs:**

- 504 Error when downloading metrics for long runs [\#61](https://github.com/mlbench/mlbench/issues/61)

**Closed issues:**

- small doc improvements for first release [\#54](https://github.com/mlbench/mlbench/issues/54)
- Check mlbench works on Google Cloud [\#51](https://github.com/mlbench/mlbench/issues/51)
- learning rate scheduler [\#50](https://github.com/mlbench/mlbench/issues/50)
- Add Nvidia k8s-device-plugin to charts [\#48](https://github.com/mlbench/mlbench/issues/48)
- Add Weave to Helm Chart [\#41](https://github.com/mlbench/mlbench/issues/41)
- Allow limiting of resources for experiments [\#39](https://github.com/mlbench/mlbench/issues/39)
- Allow downloading of Run measurements [\#35](https://github.com/mlbench/mlbench/issues/35)
- Worker Details page [\#33](https://github.com/mlbench/mlbench/issues/33)
- Run Visualizations [\#32](https://github.com/mlbench/mlbench/issues/32)
- Show experiment history in Dashboard [\#18](https://github.com/mlbench/mlbench/issues/18)
- Show model progress in Dashboard [\#13](https://github.com/mlbench/mlbench/issues/13)
- Report cluster status in Dashboard [\#12](https://github.com/mlbench/mlbench/issues/12)
- Send metrics from SGD example to metrics api [\#11](https://github.com/mlbench/mlbench/issues/11)
- Add metrics endpoint for experiments [\#10](https://github.com/mlbench/mlbench/issues/10)
- Let Coordinator Dashboard start a distributed Experiment [\#9](https://github.com/mlbench/mlbench/issues/9)
- Add mini-batch SGD model experiment [\#8](https://github.com/mlbench/mlbench/issues/8)
- add benchmark code for MPI [\#7](https://github.com/mlbench/mlbench/issues/7)
- add benchmark code for tensorflow [\#6](https://github.com/mlbench/mlbench/issues/6)
- add benchmark code for apache reef [\#5](https://github.com/mlbench/mlbench/issues/5)
- add benchmark code for apache flink [\#4](https://github.com/mlbench/mlbench/issues/4)
- get initial benchmark numbers \(spark reference implementation and mllib/ml\) [\#3](https://github.com/mlbench/mlbench/issues/3)
- evaluate script \(framework-independent\) and algorithm output format [\#2](https://github.com/mlbench/mlbench/issues/2)
- bench-spark: remove prepare-data for now, comment on solver prequisites [\#1](https://github.com/mlbench/mlbench/issues/1)



\* *This Change Log was automatically generated by [github_changelog_generator](https://github.com/skywinder/Github-Changelog-Generator)*

\* *This Change Log was automatically generated by [github_changelog_generator](https://github.com/skywinder/Github-Changelog-Generator)*

\* *This Change Log was automatically generated by [github_changelog_generator](https://github.com/skywinder/Github-Changelog-Generator)*

\* *This Change Log was automatically generated by [github_changelog_generator](https://github.com/skywinder/Github-Changelog-Generator)*

\* *This Change Log was automatically generated by [github_changelog_generator](https://github.com/skywinder/Github-Changelog-Generator)*

\* *This Change Log was automatically generated by [github_changelog_generator](https://github.com/skywinder/Github-Changelog-Generator)*

\* *This Change Log was automatically generated by [github_changelog_generator](https://github.com/skywinder/Github-Changelog-Generator)*

\* *This Change Log was automatically generated by [github_changelog_generator](https://github.com/skywinder/Github-Changelog-Generator)*

\* *This Change Log was automatically generated by [github_changelog_generator](https://github.com/skywinder/Github-Changelog-Generator)*

\* *This Change Log was automatically generated by [github_changelog_generator](https://github.com/skywinder/Github-Changelog-Generator)*

\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*


\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*


\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
