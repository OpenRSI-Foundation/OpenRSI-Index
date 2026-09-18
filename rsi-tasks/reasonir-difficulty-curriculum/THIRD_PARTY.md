# Third-party notices

This package retrieves immutable public sources and artifacts during its image build and redistributes 24 fixed BRIGHT Parquet assets under `tests/assets`. The four BEIR archives and two BRIGHT document files (Earth Science and Psychology) are not redistributed in this repository; the task setup downloads the exact upstream files before Harness runs. Those two BRIGHT files contain third-party credential-like strings in the original crawled webpage text. They remain unchanged upstream benchmark content, not credentials used by this task; setup verifies their pinned hashes without publishing the files in Git.

This notice records the terms stated by the upstream sources; it does not grant
additional rights. The SHA-256 digests of the redistributed or locally fetched evaluation files are recorded in
`tests/assets/manifest.json`.

- ReasonIR: `facebookresearch/ReasonIR@0aac96269e455965949df16520fab72da68ffc22`; see the upstream repository license.
- ReasonIR-8B and reasonir-data: `reasonir` Hugging Face repositories at revisions listed in `environment/asset-lock.json`; consult their model/data cards and repository metadata.
- BRIGHT: `xlangai/BRIGHT@3066d29c9651a576c8aba4832d249807b181ecae`. The pinned [dataset card](https://huggingface.co/datasets/xlangai/BRIGHT/blob/3066d29c9651a576c8aba4832d249807b181ecae/README.md) declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite Hongjin Su et al., *BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval*.
- GritLM: `ContextualAI/gritlm@971068105a8508bca421841c59fddba7f6596402`; Apache-2.0.
- BEIR framework: `beir-cellar/beir@6ef8c9097ebfb203ad360bd64e0cfb93e64f4a44`; its code is Apache-2.0. The pinned [BEIR README](https://github.com/beir-cellar/beir/blob/6ef8c9097ebfb203ad360bd64e0cfb93e64f4a44/README.md#disclaimer) expressly says that BEIR's redistribution does not establish a right to use its packaged datasets. Apache-2.0 applies to the framework code, not to the four separately governed datasets below.

## Fetched BEIR datasets

Task setup fetches the exact upstream BEIR corpus/query/qrels archives and verifies them against `tests/assets/manifest.json`. The archives are ignored local prerequisites, not repository content. Downloading or using them remains subject to each source's terms.

- ArguAna: the original *ArguAna Counterargs* record by Henning Wachsmuth, Shahbaz Syed, and Benno Stein declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see [Zenodo DOI 10.5281/zenodo.3973258](https://doi.org/10.5281/zenodo.3973258). The fetched archive is BEIR's corpus/query/qrels reformulation.
- SciFact: cite David Wadden et al., *Fact or Fiction: Verifying Scientific Claims*. The upstream [license notice](https://github.com/allenai/scifact/blob/master/LICENSE.md) licenses claims and evidence annotations under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and the Semantic Scholar/S2ORC abstracts in the corpus under [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/). The fetched archive is BEIR's corpus/query/qrels reformulation.
- NFCorpus: cite Vera Boteva, Demian Gholipour, Artem Sokolov, and Stefan Riezler, *A Full-Text Learning to Rank Dataset for Medical Information Retrieval*. The primary [NFCorpus terms](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) state that the corpus is free to use for academic purposes and direct other uses of included NutritionFacts.org data to its author. They do not state a general redistribution license; users must determine that their download and use comply with those terms.
- FiQA-2018: cite Macedo Maia et al., *WWW'18 Open Challenge: Financial Opinion Mining and Question Answering*, [DOI 10.1145/3184558.3192301](https://doi.org/10.1145/3184558.3192301). The primary [FiQA site](https://sites.google.com/view/fiqa/) states that both training and testing data are available only for non-commercial use and does not state a general redistribution license; users must determine that their download and use comply with those terms.

The task does not change benchmark record contents. BEIR supplies its four datasets in a normalized corpus/query/qrels layout. The large LeetCode BRIGHT document parquet is deterministically split by row into native parquet shards solely for repository transport; this transport-only change is disclosed for CC BY attribution purposes.
