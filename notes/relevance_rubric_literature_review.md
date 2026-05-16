# Designing Relevance Rubrics for (Query, Chunk) Judgments: A Literature Review

**Scope.** This review surveys how researchers and practitioners have designed rubrics for judging the relevance of a passage/chunk to a query. It is written to inform a design choice for a RAG system targeting nurses and midwives, where an LLM judge will score retrieved chunks. The motivating question: should we keep the current 3-level rubric (`D1 × (D2 + D3) ∈ {0,1,2}`) or move to a finer-grained 0–5 graded scale?

The review proceeds in seven sections — classical IR, multi-dimensional relevance theory, medical/clinical IR, QA/passage-retrieval benchmarks, RAG evaluation frameworks, LLM-as-judge literature, and a synthesis. Citations are inline (URLs). Where a specific claim could not be verified from a primary source within the time budget, it is marked `[unverified]`.

---

## 1. Classical IR: Cranfield, TREC, and graded relevance

**Cranfield (1958–1966).** Cleverdon's experiments established the modern IR evaluation paradigm of (a) a fixed test collection, (b) a set of topics, (c) human relevance judgments, and (d) precision/recall as the metrics. Initially, "relevance" meant a known-target document; Cranfield 2 moved to 3rd-party judgments on whether retrieved items addressed a query — i.e., relevance as a *judgment*, not identity. See [Cranfield experiments (Wikipedia)](https://en.wikipedia.org/wiki/Cranfield_experiments) and Robertson's [history of IR evaluation](https://www.staff.city.ac.uk/~sbrp622/papers/JIS_history_preprint.pdf).

**TREC's binary tradition.** For most of TREC's history, NIST assessors used a binary judgment. The operational definition is famous and worth quoting:

> "If you were writing a report on the subject of the topic and would use the information contained in the document in the report, then the document is relevant."

This single-sentence anchor doubles as a *task framing*, not just a definition — it tells the assessor which use-case to imagine. See the [TREC how-to](https://trec.nist.gov/howto.html) and [Overview of TREC 2002](https://trec.nist.gov/pubs/trec11/papers/OVERVIEW.11.pdf). NIST assessors are typically retired intelligence analysts who receive track-specific training.

**Shift to graded relevance.** Over the 2000s and 2010s, TREC tracks increasingly moved to graded scales. A typical TREC scheme is the **4-point** scale: 0 = Not relevant, 1 = Relevant, 2 = Highly relevant, 3 = Perfectly relevant; levels ≥1 are treated as binary-relevant for older metrics. See [arXiv:1903.11272 on graded relevance](https://arxiv.org/pdf/1903.11272) and the Springer chapter [Graded Relevance](https://link.springer.com/chapter/10.1007/978-981-15-5554-1_1).

**Sormunen (2002, SIGIR).** A widely cited reassessment of TREC-7 and TREC-8 pools introduced a 4-level scale: (0) irrelevant, (1) marginally relevant — only points to the topic, (2) fairly relevant — topical but not exhaustive, (3) highly relevant — exhaustive on the topic. Sormunen's empirical finding was that ~50% of documents originally labeled "relevant" were actually marginal under his criteria, and only ~16% were highly relevant. This kicked off the "graded relevance" research agenda. See [Liberal relevance criteria of TREC](https://dl.acm.org/doi/10.1145/564376.564433) and the [ResearchGate PDF](https://www.researchgate.net/publication/2543785_Liberal_Relevance_Criteria_of_TREC-_Counting_on_Negligible_Documents).

**Binary vs graded — does it change system rankings?** Kekäläinen's 2005 study [Binary and graded relevance in IR evaluations](https://www.sciencedirect.com/science/article/abs/pii/S0306457305000075) compared the effect on system rankings: graded relevance amplifies differences between systems, especially when nDCG-style measures are used. This is the standard argument for why TREC's modern Deep Learning Track uses a 0–3 graded scale.

**TREC Deep Learning Track (2019–2023).** Adopted a 0–3 scale that has become the de facto standard in retrieval evaluation:
- 0 = Irrelevant
- 1 = Related (but doesn't answer)
- 2 = Highly relevant (answer present but maybe unclear / hidden)
- 3 = Perfectly relevant (passage dedicated to the query, contains exact answer)

The wording is from [UMBRELA (arXiv:2406.06519)](https://arxiv.org/html/2406.06519v1), which exactly reproduces the TREC DL prompt.

**Pooling.** The rubric must coexist with the *pooling protocol*: only documents in the union of top-K results across systems are judged. Unjudged = not relevant. Pooling biases new systems (which surface unjudged docs) — see [Overview of TREC 2023](https://trec.nist.gov/pubs/trec32/papers/overview_32.pdf).

---

## 2. Multi-dimensional relevance theory

**Saracevic (1996, 2007).** The canonical multi-dimensional view of relevance distinguishes five manifestations:
1. **Algorithmic/system relevance** — query–document term/feature match (the only thing systems "see").
2. **Topical relevance** — aboutness; does the document discuss the topic.
3. **Cognitive relevance / pertinence** — meaningfulness in light of the user's knowledge state.
4. **Situational relevance / utility** — usefulness for the task at hand.
5. **Motivational/affective relevance** — relation to intentions, goals, emotions.

See Saracevic's Part II [JASIST 2007 paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/asi.20682) and the Academia.edu copy of [Part II](https://www.academia.edu/2820774/). For midwifery RAG, all five operate: TREC-style "topical" answers only the first two; "situational/utility" is the dimension that matters most for clinical decisions.

**Borlund's IIR framework.** Borlund (2003) extended evaluation to *interactive IR* via the concept of a **simulated work task situation** — a short cover story describing the scenario that prompts an information need. Situational relevance is judged against this scenario, not just the query. See [The concept of relevance in IR (Borlund 2003)](https://onlinelibrary.wiley.com/doi/abs/10.1002/asi.10286) and [The IIR evaluation model](https://informationr.net/ir/8-3/paper152.html). For a clinical RAG product, the analog is: judge relevance against the *clinical scenario the query implies*, not just the literal query string.

**Dimensions of relevance (Schamber et al., Mizzaro).** A long line of work — Mizzaro 1997, Schamber 1990s — catalogues criteria users actually report when judging relevance: topicality, novelty, recency, source authority, presentation, etc. Modern multi-dimensional rubrics (CLEF eHealth, TREC Health Misinfo) are direct descendants.

---

## 3. Medical / clinical IR rubrics

### 3.1 TREC Clinical Decision Support (CDS), 2014–2016

**Setup.** Topics were full case descriptions (chief complaint, history, exam, tests). Systems retrieved PubMed/PMC full-text articles. Medical librarians and physicians trained in informatics judged a pool of articles per topic. See [TREC CDS overview page](https://trec.nist.gov/data/clinical.html), [TREC 2016 CDS overview PDF](https://trec.nist.gov/pubs/trec25/papers/Overview-CL.pdf), [TREC 2014 CDS data](https://trec.nist.gov/data/clinical2014.html), and the survey [State-of-the-art in biomedical literature retrieval for clinical cases (Roberts et al.)](https://link.springer.com/article/10.1007/s10791-015-9259-x).

**Scale.** A 3-level graded scale:
- 0 = Not relevant
- 1 = Possibly/partially relevant
- 2 = Definitely relevant

For infNDCG, weights 0/1/2 were used directly. The scale was applied per **task type**: each topic was tagged as diagnosis, test, or treatment — meaning the rubric was *implicitly conditioned on intent*.

**Dataset size.** ~37,000 judgments per year over 30 topics — a useful reference point for what trained medical assessors can do at scale.

### 3.2 TREC Precision Medicine (PM), 2017–2020

**Multi-faceted assessment.** PM's rubric is the most sophisticated medical-IR rubric in the literature. Assessors first assigned **multi-class labels along four dimensions** (Disease, Gene, Demographic, Other) and the system *automatically converted* these to a 0/1/2 score. See the [TREC 2017 PM relevance guidelines PDF](http://www.trec-cds.org/relevance_guidelines.pdf) and [PMC overview](https://pmc.ncbi.nlm.nih.gov/articles/PMC7410346/).

Dimensions and their categorical levels:
- **Disease** ∈ {Exact, More Specific, More General, Not Disease}
- **Gene** (per gene) ∈ {Exact, Missing Variant, Different Variant, Missing Gene}
- **Demographic** ∈ {Matches, Excludes, Not Discussed}
- **Other** (comorbidities/factors) ∈ {Matches, Excludes, Not Discussed}

**Aggregation rule** (deterministic):
- **Definitely Relevant** (score 2): Disease ∈ {Exact, More Specific}, ≥1 Gene Exact, Demographic & Other ∈ {Matches, Not Discussed}.
- **Partially Relevant** (score 1): looser conditions — allows More General disease and Missing/Different variants.
- **Not Relevant** (score 0): neither of the above.

A pre-filter ("Human PM" / "Animal PM" / "Not PM") triages obviously off-topic content before detailed assessment.

**Why this matters for us.** PM is an extremely close analog to our `D1 × (D2 + D3)` design: it has a **gating dimension** (the PM screen, analogous to D1) and **substantive dimensions** combined deterministically (D2 + D3 analogs). This is strong precedent for keeping a multi-dimensional structure with a hand-designed aggregator.

### 3.3 TREC Health Misinformation (2020–2022)

Health Misinfo went further with **three independent dimensions**: usefulness, correctness (factual accuracy), and credibility (source trustworthiness). Credibility evolved from binary (2020) to a 3-level scale (0 = low, 1 = good, 2 = excellent). See [TREC 2020 Health Misinformation overview](https://trec.nist.gov/pubs/trec29/papers/OVERVIEW.HM.pdf), [TREC 2021 assessing guidelines](https://trec-health-misinfo.github.io/docs/TREC-2021-Health-Misinformation-Track-Assessing-Guidelines_Version-2.pdf), and [TREC 2022 guidelines](https://trec-health-misinfo.github.io/docs/TREC-2022-Health-Misinformation-Track-Assessing-Guidelines.pdf).

**Key insight.** Health information requires going beyond topic match because *a topically-relevant document can be wrong*. For a midwifery RAG system, this is directly relevant: a passage may mention the symptoms in the query but recommend a non-evidence-based intervention.

### 3.4 CLEF eHealth IR (2013–2021)

CLEF eHealth's consumer-health track introduced **multidimensional relevance assessments**: topical relevance, understandability, and trustworthiness. Metrics like uRBP and u+tRBP weight RBP by these orthogonal axes. See [CLEF 2017 IR task overview](https://ceur-ws.org/Vol-1866/invited_paper_16.pdf), [2016 overview](https://ceur-ws.org/Vol-1609/16090015.pdf), [CLEF eHealth task evaluation package](http://clefpackages.elra.info/clefehealthtask3/guidelines/index.html), and [2021 overview](https://hal.science/hal-03369846/file/CLEF_eHealth_21___LNCS_Overview.pdf).

This is conceptually very close to the situation in a midwifery product: nurses/midwives need *usable* (understandable, jargon-appropriate) and *trustworthy* information, not just topical matches.

### 3.5 BioASQ

BioASQ-b (snippet retrieval) uses **mean F-measure over character-level overlap** since BioASQ9 — i.e., snippet relevance is essentially *binary at the span level*, not a graded judgment. See [BioASQ FAQ](https://participants-area.bioasq.org/faq/) and [BioASQ 2025 overview](https://arxiv.org/pdf/2508.20554). This is a fundamentally different design — extraction, not grading — and informs us that for fine-grained citation-style RAG one can side-step the rubric question entirely by working at the span level.

### 3.6 MedRAG / MIRAGE

Modern medical RAG benchmark. MIRAGE (Xiong et al., 2024) consists of 7,663 medical QA questions from 5 datasets and evaluates **end-to-end answer accuracy** (multiple-choice/exact-match), not (query, chunk) relevance directly. See [Benchmarking Retrieval-Augmented Generation for Medicine (ACL 2024)](https://aclanthology.org/2024.findings-acl.372/) and [arXiv:2402.13178](https://arxiv.org/abs/2402.13178). Gap: MedRAG does not publish chunk-level relevance labels, so the medical-RAG community has no analog of the TREC PM rubric for the (query, retrieved-chunk) granularity. [unverified] for whether any subset is released with chunk labels.

### 3.7 Evidence pyramid

Medical IR uniquely cares about the **level of evidence**: systematic reviews and meta-analyses sit at the top, then RCTs, then cohort studies, etc. See [Hierarchy of evidence (Wikipedia)](https://en.wikipedia.org/wiki/Hierarchy_of_evidence) and [PMC: Levels of Evidence](https://pmc.ncbi.nlm.nih.gov/articles/PMC12064251/). Some clinical IR rubrics implicitly encode this by preferring higher-quality study designs.

---

## 4. QA and passage-retrieval benchmarks

**MS MARCO.** Originally a single-positive-passage binary scheme (one labelled relevant passage per query from Bing logs). Shallow (typically 1–2 relevance judgments per query) — see [MS MARCO benchmarking](https://arxiv.org/pdf/2105.04021). Re-annotation efforts (e.g., master's thesis [Re-ranking BERT on MS MARCO](https://tomjg14.github.io/Master_Thesis_MSMARCO_Passage_Reranking_BERT/)) added 1–5 graded labels and found this changes the picture substantially. The MS MARCO binary scheme is widely criticized as too coarse for modern reranker evaluation.

**Natural Questions.** Annotators select a **long answer** (a paragraph/table containing the answer) and a **short answer** (entity span). Effectively binary at the long-answer level; null is a valid label. Precision of annotation: 90% long / 84% short. See [NQ paper (TACL)](https://aclanthology.org/Q19-1026.pdf). This is span-extraction, not graded relevance.

**BEIR.** A *heterogeneous* benchmark — datasets carry whatever labels they originally had (mostly binary; some graded). nDCG@10 is the unifying metric. See [BEIR (NeurIPS 2021)](https://arxiv.org/abs/2104.08663) and [github.com/beir-cellar/beir](https://github.com/beir-cellar/beir). The lesson: there is no single "right" scale; what matters is that the metric is robust to the scale choice.

**TREC RAG 2024 / AutoNuggetizer.** A different paradigm: relevance is judged not at the chunk level but at the **nugget level**. A "nugget" is a one-sentence atomic fact needed to answer the topic; nuggets are tagged vital/okay; system answers are scored by how many nuggets they support (fully/partial/not). See [TREC 2024 RAG track guidelines](https://trec-rag.github.io/annoucements/2024-track-guidelines/) and [AutoNuggetizer paper (arXiv:2411.09607)](https://arxiv.org/abs/2411.09607). This is the most influential recent reformulation: it sidesteps the "is this chunk relevant?" question by asking instead "which facts does this answer cover?".

---

## 5. RAG evaluation frameworks (2023–2025)

**RAGAS — Context Precision / Context Relevance.** RAGAS asks an LLM to judge each retrieved chunk's relevance to the query, then computes a precision@k over the retrieved list. The underlying chunk judgment is binary (relevant or not) but the aggregate is a [0,1] score. See [RAGAS Context Precision docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/) and the [RAGAS metrics overview](https://docs.ragas.io/en/stable/concepts/metrics/overview/). Several variants exist (LLM-based with reference, LLM-based with response, similarity-based).

**TruLens / RAG Triad.** Three metrics: context relevance, groundedness, answer relevance. Each is computed by an LLM judge using a configurable rubric — often 0–10 or 0–3. See [TruLens RAG Triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/) and Snowflake's [Benchmarking LLM-as-a-Judge for the RAG Triad](https://www.snowflake.com/en/engineering-blog/benchmarking-LLM-as-a-judge-RAG-triad-metrics/).

**ARES (Saad-Falcon et al., NAACL 2024).** Generates synthetic (query, chunk, answer) triples, fine-tunes lightweight LM judges on each of three criteria — context relevance, answer faithfulness, answer relevance — and uses prediction-powered inference (PPI) to combine judge predictions with a small human-labelled validation set for confidence intervals. See [ARES (arXiv:2311.09476)](https://arxiv.org/abs/2311.09476) and [ARES NAACL paper](https://aclanthology.org/2024.naacl-long.20/). Important methodological contribution: the rubric for context relevance is a *trained* classifier, not a prompted scale; this sidesteps the "what number of levels?" question.

**RAGTruth (Niu et al., ACL 2024).** Word-span-level hallucination annotations: ~18K examples, 4 hallucination types (evident conflict, subtle conflict, evident baseless, subtle baseless). No separate passage-relevance rubric — it's purely about response faithfulness given retrieval. See [RAGTruth (arXiv:2401.00396)](https://arxiv.org/html/2401.00396v1) and [ACL 2024 paper](https://aclanthology.org/2024.acl-long.585.pdf).

**CRAG (Meta AI, NeurIPS 2024).** End-to-end RAG benchmark with a 4-class **answer scoring** rubric (perfect / acceptable / missing / incorrect) and a scoring rule that penalizes hallucination harder than missing answers (+1 / 0 / −1). See [CRAG paper (arXiv:2406.04744)](https://arxiv.org/abs/2406.04744). CRAG's contribution is at the *answer* level, not at the (query, chunk) level — relevant only because the missing/incorrect distinction shows how to penalize confident-but-wrong content, a critical concern in clinical settings.

---

## 6. LLM-as-judge rubric design (2023–2025)

### 6.1 Foundational papers

**Zheng et al. (NeurIPS 2023) — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena."** GPT-4 reaches ~80% agreement with humans — comparable to human–human agreement. Identifies position bias, verbosity bias, self-enhancement bias, and limited reasoning. See [arXiv:2306.05685](https://arxiv.org/abs/2306.05685). This is the foundational work that legitimized LLM-as-judge.

**Faggioli et al. (ICTIR 2023) — "Perspectives on LLMs for Relevance Judgment."** A spectrum from manual to fully automated judgments, with pilots showing reasonable correlation with humans but caveats. See [arXiv:2304.09161](https://arxiv.org/pdf/2304.09161) and [ACM page](https://dl.acm.org/doi/10.1145/3578337.3605136).

**Thomas et al. (SIGIR 2024) — "Large Language Models Can Accurately Predict Searcher Preferences."** Microsoft Bing's internal practice: GPT-4 with a 4-point relevance scale (0/1/2/3, identical wording to TREC DL) is used at production scale for ranker development. Reports human-level agreement on system rankings. [Microsoft Research PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/09/LLMs_for_relevance_labelling__SIGIR_24_-2.pdf).

**UMBRELA (Upadhyay et al., 2024).** Open-source reproduction of the Bing approach. Confirms 0–3 scale, demonstrates high correlation with human-derived system rankings across TREC DL 2019–2023. Importantly, finds LLMs are **more lenient than humans**: in re-judged TREC DL data, LLMs labeled ~26% more documents as "perfectly relevant" than humans, and humans labeled ~13% more as non-relevant. See [arXiv:2406.06519](https://arxiv.org/html/2406.06519v1) and [UMBRELA SIGIR 2025 large-scale study](https://cs.uwaterloo.ca/~jimmylin/publications/3731120.3744605.pdf).

### 6.2 Scale design — what number of levels is best?

**Empirical study: 0–5 is best on average.** A 2026 study ["Grading Scale Impact on LLM-as-a-Judge"](https://arxiv.org/html/2601.03444v1) compared 0–5, 0–10, and 0–100 on subjective and objective tasks. Aggregated across tasks, 0–5 had the best human–LLM ICC (0.853) and lowest nMAE (0.111); 0–10 was consistently the weakest. The paper did not test binary or 0–3, however.

**Criteria-based decomposition.** Farzi & Dietz (2025) ["Criteria-Based LLM Relevance Judgments"](https://arxiv.org/html/2507.09488) decompose relevance into four sub-criteria, each on a 0–3 scale:
- **Exactness**: How precisely does the passage answer the query?
- **Topicality**: Is the passage about the same subject as the whole query?
- **Coverage**: How much of the passage is dedicated to the query and related topics?
- **Contextual Fit**: Does the passage provide relevant background/context?

Sub-criteria are aggregated either by (a) feeding all four grades into another LLM call for a final 0–3 label, or (b) **sumdecompose**: sum the four 0–3 scores (range 0–12) and threshold (10–12 → 3, 7–9 → 2, 5–6 → 1, 0–4 → 0). This is the closest published analog to our `D1 × (D2 + D3)` design and validates the multi-dimensional approach.

**LLMJudge SIGIR 2024 challenge.** 42 LLM-generated label sets from 8 teams. Some teams used 4-graded scales, others binary. LLMs tend to be more lenient on graded scales; converting to binary with a threshold often recovers human-correlated rankings. See [LLMJudge paper (arXiv:2408.08896)](https://arxiv.org/pdf/2408.08896) and the benchmark study ["Judging the Judges" (arXiv:2502.13908)](https://arxiv.org/pdf/2502.13908).

**Binary vs graded for LLMs — recent findings.** [Arabzadeh & Clarke (arXiv:2504.12558)](https://arxiv.org/pdf/2504.12558) compares binary, graded, pairwise-preference, and nugget-based LLM judgments. Preference judgments correlate most with humans on individual labels; binary and graded both produce strong *system-ranking* correlations. The takeaway: for ranking systems (Kendall's tau), graded vs binary matters less than expected; for *labeling individual chunks* (per-item agreement), pairwise preference is best but expensive.

### 6.3 Holistic vs analytic rubrics

[Adnan Masood's overview (Medium, 2026)](https://medium.com/@adnanmasood/rubric-based-evals-llm-as-a-judge-methodologies-and-empirical-validation-in-domain-context-71936b989e80) summarizes the field: holistic rubrics (one overall score) are faster and yield higher single-score agreement, but obscure failure modes. Analytic rubrics (separate criteria) are easier to debug but can show larger LLM–human gaps on individual sub-traits. The literature increasingly favors *analytic* designs for high-stakes domains (CheckEval, RocketEval, FLASK, Criteria-Based LLM Relevance Judgments). See also [LLM Essay Scoring (arXiv:2604.00259)](https://arxiv.org/html/2604.00259) for an empirical comparison.

### 6.4 Bias and consistency

- **Position bias** in pairwise: Shi et al. ["Judging the Judges: Position Bias" (arXiv:2406.07791)](https://arxiv.org/abs/2406.07791) — non-random, varies by judge model and quality gap.
- **Verbosity / length bias**: ["Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge" (arXiv:2410.02736)](https://arxiv.org/html/2410.02736v1) — judges prefer longer outputs; can be mitigated with structured rubrics.
- **Self-preference bias**: ["Self-Preference Bias in LLM-as-a-Judge" (arXiv:2410.21819)](https://arxiv.org/html/2410.21819v2) — judges score their own model's outputs higher.
- **Pairwise vs pointwise vulnerability**: ["Pairwise or Pointwise?" (arXiv:2504.14716)](https://arxiv.org/abs/2504.14716) — pairwise preferences flip ~35% of the time under perturbation, vs ~9% for absolute scores. So pairwise is *not* automatically more reliable.

### 6.5 Best practices (consensus)

From [Aman's primer](https://aman.ai/primers/ai/LLM-as-a-judge/), [Evidently AI's guide](https://www.evidentlyai.com/llm-guide/llm-as-a-judge), and surveys like [arXiv:2411.15594](https://arxiv.org/html/2411.15594v6):

1. **Anchor each level with a worded definition** — not just a number. A 5-point scale without level-by-level descriptors regresses to noise.
2. **Provide 1–2 worked examples per level** (calibration shots).
3. **Force chain-of-thought before the score** — improves consistency, especially on multi-criterion rubrics.
4. **Structured output** (JSON with reasoning + score).
5. **Validate on a 30–50-item human-labelled gold set** and report ICC / Cohen's kappa.
6. **Use multiple judge models or prompt paraphrases** to detect prompt sensitivity.
7. **Convert to binary at metric time if needed** — keeping graded labels gives flexibility downstream.

---

## 7. Synthesis

### 7.1 What scales are most common?

| Tradition | Typical scale | Notes |
|---|---|---|
| TREC ad-hoc (legacy) | Binary | "Would you use it in a report?" anchor |
| TREC DL / UMBRELA / Bing | **0–3 (4-point)** | Irrelevant / Related / Highly / Perfectly |
| Sormunen / SIGIR graded relevance | 0–3 (4-point) | Irrelevant / Marginal / Fair / Highly |
| TREC CDS (medical) | **0–1–2 (3-point)** | Not / Partially / Definitely relevant |
| TREC Precision Medicine | Multi-dim → 0/1/2 | Cascade: categorical dimensions, deterministic aggregator |
| TREC Health Misinfo | 3 separate axes | Usefulness × correctness × credibility |
| CLEF eHealth | 3 separate axes | Topicality × understandability × trustworthiness |
| MS MARCO | Binary | Single positive per query |
| BEIR | Heterogeneous | Inherits per dataset |
| RAGAS context precision | Binary per chunk → continuous aggregate | LLM judge |
| TruLens context relevance | Often 0–10 or 0–1 | Configurable |
| LLM-as-judge (general) | 1–5 most common; 0–5 empirically best | [arXiv:2601.03444](https://arxiv.org/html/2601.03444v1) |
| Modern LLM relevance papers | 0–3 dominant | Mirrors TREC DL |

**The two dominant choices in modern IR/RAG work are 0–3 (passage relevance) and 0–5 (open-ended LLM judging).** Medical IR has historically used 3-point (0/1/2) because medical assessors find it hard to discriminate between 4+ ordinal levels reliably ([CDS overview](https://trec.nist.gov/pubs/trec25/papers/Overview-CL.pdf), [unverified] for the explicit "why" — but inter-rater agreement studies in clinical annotation generally support this).

### 7.2 Single-dimensional vs multi-dimensional

There is a clear *re-emergence* of multi-dimensional designs in 2024–2025 LLM-as-judge work, after a decade in which single-axis graded scales dominated. The drivers:

- **Debuggability.** Analytic rubrics let you locate where the judge (or the system) is failing.
- **Calibration.** Sub-scores act as anchors; the LLM produces them via CoT, then an aggregator turns them into the final label.
- **Domain fit.** Clinical/legal/financial domains have orthogonal concerns that don't reduce to "topicality."

Strong precedents for multi-dimensional rubrics with deterministic aggregators:
- **TREC Precision Medicine** (Disease × Gene × Demographic × Other → 0/1/2)
- **TREC Health Misinformation** (Usefulness × Correctness × Credibility)
- **CLEF eHealth** (Topicality × Understandability × Trustworthiness)
- **Criteria-Based LLM Relevance Judgments** (Exactness × Topicality × Coverage × Contextual Fit)

Our current `D1 × (D2 + D3)` design is squarely in this tradition.

### 7.3 What's specific to medical/clinical relevance?

Beyond topical match, clinical IR rubrics consistently add:

1. **Actionability / clinical utility** — does the passage say what to *do*? (TREC CDS task-type tagging; CLEF eHealth utility framing.)
2. **Specificity to the clinical scenario** — patient demographics, comorbidities, history. (TREC PM's Disease/Gene/Demographic/Other.)
3. **Evidence level** — is this a guideline, RCT, case report, or anecdote? ([Hierarchy of evidence](https://en.wikipedia.org/wiki/Hierarchy_of_evidence)) — usually applied via document-type metadata, not the rubric.
4. **Correctness / agreement with current standard of care** — distinct from topicality. (TREC Health Misinfo correctness axis.)
5. **Credibility / source trustworthiness** — guideline body, peer-reviewed journal, etc. (Health Misinfo, CLEF eHealth.)
6. **Understandability / appropriate register** — for consumer or non-specialist audiences. (CLEF eHealth.)

For a **midwifery** RAG used by nurses/midwives (not consumers), understandability is less critical, but actionability, scenario-specificity, and correctness are paramount.

### 7.4 What rubric properties produce more consistent LLM judgments?

Synthesizing across the bias and consistency literature:

- **Fewer, well-anchored levels beat many under-defined levels.** Empirically 0–3 to 0–5 is the sweet spot; 0–10 and 0–100 produce noise.
- **Worded anchors per level are non-negotiable.** Without descriptors, LLMs cluster around the middle.
- **CoT / decomposition before scoring improves stability.** Multi-criterion analytic rubrics with CoT outperform single holistic prompts on debugging *and* stability.
- **Pairwise is more aligned per-instance but less stable under perturbation.** Absolute scoring with a structured rubric is the safer default for batch evaluation.
- **Graded scales are not harder than binary for LLMs *on average*, but LLMs are more lenient on graded scales** — they over-assign the top label. Calibration on a human-labelled gold set is needed before trusting absolute numbers; ranking correlations are more robust.
- **Length and verbosity bias affect chunk judgment.** Long, dense chunks may receive inflated relevance scores; short snippets may be under-rated. Mitigation: include passage length as a feature the rubric explicitly tells the judge to ignore.

### 7.5 Recommendation for the midwifery RAG situation

The system retrieves 3 chunks; the question is "are the 3 chunks useful for generating a good clinical answer?" The literature points strongly to the following:

**(A) Keep multi-dimensional, but expand and re-anchor.**

A bare `D1 × (D2 + D3) ∈ {0,1,2}` is in the right *tradition* (TREC CDS, TREC PM) but is squeezed at the top: it cannot distinguish "all three chunks are good" from "two are great and one is mediocre." Three concrete options ranked by precedent strength:

1. **Keep the analytic decomposition; widen the final scale to 0–3.** Mirror TREC DL: Irrelevant / Related / Highly / Perfectly. Compute it as a deterministic function of the sub-dimensions, the way TREC PM does. The sub-dimension definitions don't need to change much.

2. **Adopt 0–5 if you want maximum LLM-as-judge alignment.** The Borji et al. 2026 study shows 0–5 has the best ICC. But 0–5 needs *six* anchor wordings, which is a real annotation-cost increase.

3. **Decompose into independent axes (Health Misinfo / CLEF style).** For midwifery: report-Topicality × Clinical-Scenario-Fit × Actionability × Evidence-Quality. Report each separately, aggregate downstream. This is the "right" thing for a high-stakes domain but is a larger rebuild.

**(B) Anchor every level with worded definitions and 1–2 worked examples.**

This is the single highest-leverage change. Use the TREC PM relevance guidelines PDF and the UMBRELA prompt as style models — both publish explicit per-level wordings.

**(C) Frame the task with a simulated work-task scenario.**

Borrowing from Borlund: tell the judge to assume "a midwife reading the 3 chunks needs to make a decision about *X*." This concretizes the situational/utility dimension and is far closer to how clinical relevance actually operates than abstract "is this passage relevant to the query."

**(D) Validate on a 30–50-item human-labelled gold set.**

This is consensus best-practice. Report ICC or quadratic-weighted kappa against expert judges (ideally midwives). Without this, scale choice is opinion.

**(E) Known pitfalls to avoid.**

- **LLM leniency.** Expect the LLM to over-assign the top label by 10–25% relative to humans (UMBRELA finding). Don't tune downstream thresholds on raw LLM scores; calibrate first.
- **Length / verbosity bias.** Long retrieved chunks may get inflated scores. Either normalize by length or instruct explicitly.
- **Self-consistency on borderline cases.** Run each chunk 3× with temperature 0.0–0.3 and either take the median or flag disagreements.
- **Position bias when multiple chunks are scored together.** Score chunks independently rather than as a 3-tuple, then aggregate; or shuffle order across runs.
- **Conflating chunk relevance with answer correctness.** RAGTruth / TruLens groundedness are *answer-side* metrics. Don't reuse the same rubric for both — the chunk-relevance judge should not see the model's answer.
- **Pooling bias** (less acute since you generate retrievals, not select from a pool — but worth noting if you evaluate multiple retrievers comparatively).

---

## Annotated bibliography (selected, with critique)

- **Sormunen (SIGIR 2002), "Liberal Relevance Criteria of TREC."** [PDF](https://www.researchgate.net/publication/2543785). Critique: post-hoc reassessment, so cannot be directly compared to original TREC ranking; small sample (38 topics).
- **Saracevic (2007), "Relevance, Part II."** [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/asi.20682). Critique: pre-LLM; the categories don't map cleanly onto LLM judge prompts.
- **Borlund (2003), "The concept of relevance in IR."** [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/asi.10286). Critique: heavily user-interactive; simulated work tasks require human users.
- **Roberts et al. (TREC CDS overviews 2014–2016).** Critique: 3-level scale chosen for assessor speed, not theoretical reasons; PM track later showed multi-dim could be done.
- **Roberts et al. (TREC PM, PMC).** [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7410346/). Critique: deterministic aggregator hides which dimension a system failed on; very domain-specific.
- **CLEF eHealth IR overviews.** [2017 PDF](https://ceur-ws.org/Vol-1866/invited_paper_16.pdf). Critique: multidimensional adds annotation cost; agreement on understandability and trustworthiness is lower than topicality.
- **TREC Health Misinformation 2021 guidelines.** [Assessing Guidelines PDF](https://trec-health-misinfo.github.io/docs/TREC-2021-Health-Misinformation-Track-Assessing-Guidelines_Version-2.pdf). Most directly relevant to clinical-correctness rubrics.
- **Faggioli et al. (ICTIR 2023), "Perspectives."** [arXiv](https://arxiv.org/pdf/2304.09161). Critique: pre-GPT-4, conclusions about LLM limitations have weakened.
- **Thomas et al. (SIGIR 2024).** [Microsoft Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/09/LLMs_for_relevance_labelling__SIGIR_24_-2.pdf). Critique: Bing's production setup; details about prompt iteration not fully disclosed.
- **Upadhyay et al., UMBRELA (2024).** [arXiv:2406.06519](https://arxiv.org/abs/2406.06519). Critique: validates Thomas et al.'s scale but inherits assumptions about TREC DL queries.
- **Farzi & Dietz, Criteria-Based LLM Relevance Judgments (2025).** [arXiv:2507.09488](https://arxiv.org/html/2507.09488). Closest precedent for our multi-dimensional design.
- **Borji et al., "Grading Scale Impact on LLM-as-a-Judge" (2026).** [arXiv:2601.03444](https://arxiv.org/html/2601.03444v1). Critique: tested 0–5, 0–10, 0–100 only; did not test 0–3 or binary. Recommends 0–5.
- **Saad-Falcon et al., ARES (NAACL 2024).** [arXiv:2311.09476](https://arxiv.org/abs/2311.09476). Critique: requires synthetic-data pipeline; lightweight judges may not generalize across domains.
- **CRAG (Yang et al., NeurIPS 2024).** [arXiv:2406.04744](https://arxiv.org/abs/2406.04744). Answer-level rubric; useful for downstream answer eval, not chunk eval.
- **Pradeep et al., AutoNuggetizer / TREC RAG 2024.** [arXiv:2411.09607](https://arxiv.org/abs/2411.09607). Strong precedent for *avoiding* chunk-level judgments altogether by working at the nugget level.
- **Zheng et al., MT-Bench (NeurIPS 2023).** [arXiv:2306.05685](https://arxiv.org/abs/2306.05685). Foundational LLM-as-judge paper.

---

## Gaps in the literature (and in this review)

- **No widely-adopted public rubric exists for medical RAG at the (query, chunk) level.** MedRAG/MIRAGE evaluate end-to-end accuracy, not chunk relevance. TREC CDS/PM rubrics target full-text articles, not chunks. Adapting requires our own anchor examples.
- **No published rubric specific to midwifery/maternal health IR** that I could verify; clinical-decision-support reviews for maternity exist (e.g., [Lancet eClinicalMedicine 2024](https://www.thelancet.com/journals/eclinm/article/PIIS2589-5370(24)00401-2/fulltext)) but discuss CDSS efficacy rather than retrieval evaluation.
- **Limited empirical work comparing 0–3 vs 0–5 vs analytic-aggregated scales on the same dataset.** Borji et al. 2026 covers 0–5/0–10/0–100; Arabzadeh & Clarke 2025 covers binary/graded/preference; no head-to-head on the three-vs-five question.
- **LLM-as-judge calibration on clinical content specifically** is thin; most published work uses general web/QA passages. [unverified] whether GPT-4-class models show different leniency on clinical passages than on Bing-style passages.
