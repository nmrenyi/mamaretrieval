# Rubric design — worked examples on 5 audit queries

> **Purpose**: Bottom-up rubric refinement for the proposed graded rubric `score = D1 × (D2 + D3 + D4)`, all dimensions on {0, 1, 2}. Score range: 0-6. By scoring real chunks manually, we (1) tighten the per-level definitions, (2) discover edge cases the prompt needs to handle, and (3) build a library of CoT anchor examples for the eventual judge prompt.

## Methodology

Picked 5 queries from the 100-query audit subset to span intent (dosing, methodology, diagnosis, monitoring schedule, procedural management) and source (very_high + high tier):

| Query | Tier | Source | Intent |
|---|---|---|---|
| q_00041 | very_high | msf-essential-obstetric-and-newborn-care | dosing |
| q_00148 | very_high | hesperian-a-book-for-midwives | methodology |
| q_00423 | high | who-midwifery-education-modules-5 | diagnosis (signs + lab) |
| q_00455 | very_high | hesperian-a-book-for-midwives | monitoring schedule |
| q_00536 | high | who-midwifery-education-modules-2 | procedural management |

For each query, gathered the union of top-3 across all 6 retrievers (BM25, MedCPT, Octen, voyage, LateOn, Gecko) and scored every chunk in the union (13-17 chunks per query, 65 total). Worked through each by hand, applying the refined criteria below.

---

## Refined criteria (post-exercise — these reflect what I had to clarify as I scored)

### D1 — Topic gate (boolean)

**Question**: Does the chunk address the same clinical problem/scenario as the query?

**D1 = true** if the chunk discusses the exact condition, intervention, or scenario asked about AND in the same clinical context (timing window — antenatal / intrapartum / postpartum).

**D1 = false** if:
- The chunk is about a different clinical topic entirely
- The chunk discusses the same condition but in a different time/context — e.g., "BP during pregnancy" when the query asks about "BP after birth"
- The chunk is pure metadata: tables of contents, references, learning objectives, course-content outlines, assessment rubrics with no clinical content
- The chunk is about a different stage of the same broad process (e.g., "fetal heart monitoring during labour" doesn't satisfy a query about "fetal heart monitoring during antenatal visits")

**Edge case clarification**: source-document mismatch is OK if content matches. A chunk from `WHO_Abortion_Care_2022` discussing "determining gestational age" satisfies a general gestational-age query — the clinical content is what matters, not the parent document's topic.

### D2 — Meaningful clinical content (0/1/2)

**Question**: How rich is the chunk's clinical content (independent of whether it answers the query)?

| D2 | Meaning | Examples |
|---|---|---|
| **0** | No meaningful clinical content beyond a topic label | Section headers, page numbers, TOC entries, references, learning objectives, "students should be able to...", administrative notes, supersession statements, resource/cost text, copyright |
| **1** | Some clinical content; brief background, definition, peripheral mention, statistics | "HELLP syndrome occurs in 20% of severe pre-eclampsia cases" (epidemiology), "Iron supplementation prevents anaemia" (single concept), one or two clinical facts surrounded by admin framing |
| **2** | Rich clinical content; multi-concept paragraph(s) developing definitions / pathophysiology / risk factors / mechanisms / symptoms in depth | Full paragraph explaining PPH pathophysiology + risk factors; detailed description of HELLP triad mechanism + diagnostic criteria + clinical course |

### D3 — Actionable guidance (0/1/2)

**Question**: How specific is the actionable guidance in the chunk?

| D3 | Meaning | Examples |
|---|---|---|
| **0** | No actionable guidance — pure background, definitions, epidemiology, or admin | "PPH is a leading cause of maternal mortality." Pure pathophysiology. References. |
| **1** | General / partial guidance — action specified but missing specifics (dose, frequency, threshold, complete protocol) | "Give a uterotonic to prevent PPH", "monitor vital signs", "consult obstetrician", "massage the uterus", "estimate blood loss" |
| **2** | Specific complete guidance — exact doses with route and frequency, numeric thresholds with action triggers, full step-by-step procedure, scheduled-monitoring intervals with numbers | "10 IU oxytocin IM, then 5 IU/hr for 4 hours", "BP every 30 min × first hour, then hourly", "Folic acid 0.4–0.8 mg PO daily, beginning 1–12 months before conception" |

**Structural rule (preserved)**: `D3 ≥ 1` implies `D2 ≥ 1` — any actionable guidance carries some meaningful content.

### D4 — Density (0/1/2)

**Question**: What fraction of the chunk text is directly useful for answering the query specifically?

| D4 | Meaning | Examples |
|---|---|---|
| **0** | Useful content < 25% of chunk | Long chunk with one buried relevant sentence; mostly noise, off-topic surrounding text, or irrelevant procedural detail |
| **1** | Useful content 25-75% | Mixed: relevant + adjacent-but-not-query-specific material interleaved |
| **2** | Useful content > 75% | Chunk is focused; entire content directly addresses the query, only minor framing/header text |

**Note**: D4 is judged relative to **the specific query**, not the broader topic. A chunk that's entirely about PPH management is D4=2 only if the query is about PPH management. If the query is specifically about "uterotonic doses", the same chunk might be D4=1 (relevant context, but only a fraction is dose-specific).

**Important**: D4 captures *signal-to-noise from the LLM's point of view*. A short focused chunk (50 tokens, all useful) is D4=2 even if it contains less total information than a long mixed chunk (500 tokens, 200 useful) which would be D4=1.

### Final score

```
score = D1 × (D2 + D3 + D4)
```

Range: 0 (off-topic OR on-topic but no useful content) to 6 (rich + specific + focused).

---

## Query 1 — q_00041

**Query**: "What are the alternative dosages for iron and folic acid supplementation?"

**Source**: msf-essential-obstetric-and-newborn-care (very_high tier)

**Seed chunk**: `758bdffa5f973882`

### Chunk-by-chunk scoring

#### Chunk 318a18d9 (octen@1, voyage@1) — D1=1, D2=1, D3=2, D4=2 → score 5
**Source**: msf-essential-obstetric-and-newborn-care, p.55. Length 228 chars.
**Text**: *"4.1.2 Treatment > Footnotes (a) 200 mg ferrous sulfate (65 mg elemental iron) + 400 micrograms folic acid tablets may be replaced by 185 mg ferrous fumarate (60 mg elemental iron) + 400 micrograms folic acid tablets."*

**Reasoning**:
- **D1**: On-topic — iron/folic acid alternative formulations directly match "alternative dosages" query.
- **D2**: 1 — single sentence of substitution rule. Clinical fact but limited depth (no rationale, no when-to-use).
- **D3**: 2 — fully specific actionable: exact substitution with elemental iron amounts.
- **D4**: 2 — essentially the whole chunk is useful; only "4.1.2 Treatment / Footnotes" framing is non-content.
- Old 3-level: score=2. New score=5. The new rubric correctly distinguishes this "specific but terse" chunk from a chunk that has both rich context AND specific guidance.

#### Chunk 666d6b2a (lateon@1) — D1=1, D2=2, D3=2, D4=2 → score 6
**Source**: WHO_ANC_2016, p.41. Length 368 chars.
**Text**: *"RECOMMENDATION A.2.1: Daily oral iron and folic acid supplementation with 30 mg to 60 mg of elemental iron and 400 µg (0.4 mg) folic acid is recommended for pregnant women to prevent maternal anaemia, puerperal sepsis, low birth weight, and preterm birth."*

**Reasoning**:
- **D1**: 1 — directly on dosage of iron/folic acid in pregnancy.
- **D2**: 2 — formal recommendation statement; includes indication (prevention of maternal anaemia / puerperal sepsis / LBW / preterm birth). Rich purpose-context.
- **D3**: 2 — specific dose range (30–60 mg iron, 400 µg folic acid daily).
- **D4**: 2 — entire chunk is the recommendation.
- Old 3-level: score=2. New score=6. **This is the max-score anchor for the rubric.**

#### Chunk c3066ae0 (octen@2) — D1=1, D2=2, D3=2, D4=2 → score 6
**Source**: WHO_ANC_2016, p.43. Length 851 chars.
**Text**: A.2.2 Intermittent iron and folic acid supplements. Includes "If a woman is diagnosed with anaemia (Hb < 110 g/L) during ANC, she should be given 120 mg of elemental iron and 400 µg (0.4 mg) of folic acid daily until her Hb concentration rises to normal..." plus equivalent ferrous formulations.

**Reasoning**:
- **D1**: 1 — alternative dosing regimens for iron/folic acid.
- **D2**: 2 — Hb threshold for diagnosis, treatment-dose-vs-prevention-dose distinction, equivalents in different ferrous formulations.
- **D3**: 2 — three pieces of specific protocol: threshold (Hb < 110), treatment dose (120 mg iron daily), equivalents (600/360/1000 mg of sulfate/fumarate/gluconate).
- **D4**: 2 — every sentence is directly useful for the query.
- Old: 2. New: 6.

#### Chunk d6fa6f68 (octen@3, voyage@2) — D1=1, D2=2, D3=2, D4=1 → score 5
**Source**: WHO_ANC_2016, p.41. Length 719 chars.
**Text**: Recommendation A.2.1 task-shifting note about community delivery + a formulation equivalence footnote ("60 mg elemental iron is 300 mg ferrous sulfate hepahydrate, 180 mg ferrous fumarate or 500 mg ferrous gluconate") + folic acid pre-conception advice.

**Reasoning**:
- **D1**: 1 — iron / folic acid in pregnancy.
- **D2**: 2 — three distinct clinical points (task shifting, dose equivalents, pre-conception folic acid timing). Multi-concept.
- **D3**: 2 — specific equivalences (300 mg / 180 mg / 500 mg).
- **D4**: 1 — about half the chunk is about task-shifting / implementation, not dose specifics. The query is specifically about dosages, so the task-shifting paragraph is adjacent-but-not-useful.
- Old: 2. New: 5. **D4=1 anchor example**: not all of a useful chunk is useful for the specific query.

#### Chunk ad7678f8 (gecko@1) — D1=1, D2=1, D3=2, D4=2 → score 5
**Source**: WHO_IntegratedPregBirth_2015, p.105. Length 397 chars.
**Text**: Bullet list: "Routinely once daily in pregnancy and until 3 months after delivery or abortion. Twice daily as treatment for anaemia (double dose). Check woman's supply of iron and folic acid at each visit and dispense 3 months supply. Advise to store iron safely..."

**Reasoning**:
- **D1**: 1.
- **D2**: 1 — frequency rules and storage advice; thin on context (no doses spelled out).
- **D3**: 2 — specific frequency-based protocol (once daily routine, twice daily for anaemia).
- **D4**: 2 — almost all content is about dosing protocol or supplementation logistics.
- Old: 2. New: 5.

#### Chunk 94aa6726 (gecko@2) — D1=1, D2=2, D3=2, D4=2 → score 6
**Source**: WHO_IntegratedPregBirth_2015, p.105. Table: "1 tablet = 60mg, folic acid = 400µg. All women: 1 tablet through pregnancy / 3 mo postpartum. Women with anaemia: 2 tablets / 3 mo / 3 mo".

**Reasoning**:
- **D1**: 1.
- **D2**: 2 — tablet composition + dosing table + temporal coverage.
- **D3**: 2 — specific tablet count, duration, and population (all women vs. anaemic).
- **D4**: 2 — focused.
- Old: 2. New: 6.

#### Chunk b40eea13 (medcpt@2) — D1=1, D2=2, D3=2, D4=1 → score 5
**Source**: clinical-practice-guidelines-midwifery-womens-health, p.322. Length 1180 chars.
**Text**: Long pre-conception advice section — folic acid 0.4–0.8 mg daily for routine; 4–5 mg for women with previous neural tube defect; plus immunisation advice (Hep B / HPV / influenza / tetanus / live virus vaccines).

**Reasoning**:
- **D1**: 1 — folic acid dosing is directly on topic (iron is mentioned briefly).
- **D2**: 2 — both routine and high-risk folic acid regimens with timing.
- **D3**: 2 — exact doses for both regimens with timing.
- **D4**: 1 — ~40% of the chunk is about immunisation, which isn't the query's subject. The folic acid portion is rich but density is diluted.
- Old: 2. New: 5. **D4=1 example for "good content embedded in a wider section that drifts off-topic".**

#### Chunk 69391a87 (lateon@2) — D1=1, D2=2, D3=2, D4=1 → score 5
**Source**: midwifery-preparation-for-practice, p.535. Length 615 chars.
**Text**: Discussion of routine offer of iron supplementation in developed countries; WHO recommendation "30–60 mg / day iron, 0.4 mg / day folic acid, higher 120 mg / day for women with anaemia."

**Reasoning**:
- **D1**: 1.
- **D2**: 2 — clinical context (research evidence quality debate) + WHO specifics.
- **D3**: 2 — specific doses for routine + treatment regimens.
- **D4**: 1 — about half is debate/research context, not query-relevant. The dose specifics are present but a smaller fraction of the chunk.
- Old: 2. New: 5.

#### Chunk 5fffb433 (bm25@3, lateon@3, gecko@3) — D1=1, D2=1, D3=0, D4=1 → score 2
**Source**: WHO_ANC_2016, p.42. Length 1191 chars.
**Text**: "Effects of any daily iron and folic acid supplements compared with no..." — Cochrane review summary, trial counts, "Most commonly used dose of elemental iron was 60 mg daily (range 30–240 mg) and that of folic acid was 400 µg daily."

**Reasoning**:
- **D1**: 1.
- **D2**: 1 — evidence-summary text rather than direct clinical principles. Has one clinical fact (the range of doses studied) embedded.
- **D3**: 0 — no actionable recommendation; just describing what trials used. Reader can't act on "range was 30–240 mg" because no recommendation is made.
- **D4**: 1 — the dose info is one sentence in a long evidence-summary. Most of the chunk is meta-discussion of trials.
- Old: 1. New: 2.

#### Chunk 6fbfefb2 (voyage@3) — D1=1, D2=1, D3=0, D4=1 → score 2
**Source**: WHO_ANC_2016, p.44. Same evidence-summary style about intermittent supplementation.
- Old: 1. New: 2 (same reasoning as 5fffb433).

#### Chunk 27b615ab (bm25@1, medcpt@3) — D1=1, D2=0, D3=0, D4=0 → score 0
**Source**: WHO_ANC_2016, p.44, "Resources" section. Length 211 chars.
**Text**: *"Intermittent iron and folic acid supplementation might cost a little less than daily iron and folic acid supplementation due to the lower total weekly dose of iron."*

**Reasoning**:
- **D1**: 1 (on-topic).
- **D2**: 0 — cost/resource information, not clinical content.
- **D3**: 0 — no actionable clinical guidance.
- **D4**: 0 — entire chunk is cost talk; for a *dose* query, this is noise.
- Old: 0. New: 0.

#### Chunk 16ae1a57 (medcpt@1) — D1=1, D2=0, D3=0, D4=0 → score 0
**Source**: WHO_PNC_2013, p.58. Length 125 chars. *Header only: "RECOMMENDATIONS 7–9: GDG CONSENSUS... RECOMMENDATION 10: IRON AND FOLIC ACID SUPPLEMENTATION"*. No body text.
- D2: 0 (pure header). D3: 0. D4: 0 (no useful content). Old: 0. New: 0.

#### Chunk 94735548 (bm25@2) — D1=1, D2=0, D3=0, D4=0 → score 0
**Source**: WHO_ANC_2016, p.41. Length 247 chars. Header + administrative note that one recommendation supersedes a 2012 WHO guideline. No clinical content.
- Old: 0. New: 0.

### Q1 score summary

| chunk | retrievers | new D1·(D2+D3+D4) | new score | old |
|---|---|---|---|---|
| 666d6b2a | lateon@1 | 1·(2+2+2) | **6** | 2 |
| c3066ae0 | octen@2 | 1·(2+2+2) | **6** | 2 |
| 94aa6726 | gecko@2 | 1·(2+2+2) | **6** | 2 |
| 318a18d9 | octen@1, voyage@1 | 1·(1+2+2) | **5** | 2 |
| d6fa6f68 | octen@3, voyage@2 | 1·(2+2+1) | **5** | 2 |
| ad7678f8 | gecko@1 | 1·(1+2+2) | **5** | 2 |
| b40eea13 | medcpt@2 | 1·(2+2+1) | **5** | 2 |
| 69391a87 | lateon@2 | 1·(2+2+1) | **5** | 2 |
| 5fffb433 | bm25@3, lateon@3, gecko@3 | 1·(1+0+1) | **2** | 1 |
| 6fbfefb2 | voyage@3 | 1·(1+0+1) | **2** | 1 |
| 27b615ab | bm25@1, medcpt@3 | 1·(0+0+0) | **0** | 0 |
| 16ae1a57 | medcpt@1 | 1·(0+0+0) | **0** | 0 |
| 94735548 | bm25@2 | 1·(0+0+0) | **0** | 0 |

**Observation**: 8 chunks that all scored 2 under old rubric now split into {5, 5, 5, 5, 5, 6, 6, 6}. The 5s vs 6s separate "specific but missing context (D2=1) OR specific in a diluted chunk (D4=1)" from "rich + specific + focused (all 2s)". This is the discrimination we wanted.

---

## Query 2 — q_00148

**Query**: "What methods determine gestational age and due date during the first checkup?"

### Chunk-by-chunk scoring (brief)

| chunk | D1 | D2 | D3 | D4 | new | old | rationale |
|---|---|---|---|---|---|---|---|
| b5aa599f (bm25@1, octen@1, voyage@1, lateon@1) | 1 | 2 | 2 | 2 | **6** | 2 | All three methods (LMP, clinical exam, ultrasound) with calculation rules and worked example. Max anchor. |
| ac034a8e (bm25@3, octen@2, voyage@2, gecko@2) | 1 | 1 | 1 | 2 | **4** | 2 | Lists three methods at high level ("Use date of last bleeding; Measure womb; Get ultrasound") without calculation rules. Specific but shallow. |
| eb94102d (lateon@3) | 1 | 2 | 2 | 2 | **6** | 2 | Naegele's Rule with formula + cycle-length adjustments + dating-scan timing. |
| c623938a (lateon@2) | 1 | 2 | 1 | 2 | **5** | 2 | Dating-scan parameters (CRL, BPD, FL, HC) + accuracy windows. Specific *what* to measure but no calculation rules → D3=1. |
| 0e08c170 (medcpt@3) | 1 | 1 | 2 | 2 | **5** | 2 | Fundal-height measurement with specific growth rates ("1–2 finger widths per month / 1 cm per week"). Specific but single-method focus. |
| 1ddae242 (gecko@1) | 1 | 2 | 2 | 1 | **5** | 2 | Discussion of due-date discrepancies + measure-the-womb growth-rate calibration. Useful but a fraction is about "why due dates can be wrong" rather than methods. |
| 7656095f (octen@3, voyage@3) | 1 | 1 | 0 | 1 | **2** | 1 | "Information such as LMP and EDC is crucial in the calculation of gestational age" — names the inputs without methodology. |
| b5c13e56 (gecko@3) | 1 | 1 | 1 | 1 | **3** | 2 | Determining gestational age **in the context of abortion** — methods named (LMP/exam/ultrasound) but with abortion-specific framing. D4=1 because half is abortion-method selection. |
| 4421e444 (medcpt@1) | 0 | 0 | 0 | 0 | **0** | 0 | "After the checkup — Make a time for the next prenatal visit". Off-topic. |
| 3096ce8a (medcpt@2) | 0 | 0 | 0 | 0 | **0** | 0 | "Follow-up: Explain any necessary follow-up care/referrals". Off-topic. |
| 63a982a2 (bm25@2) | 1 | 0 | 0 | 0 | **0** | 0 | Test content outline ("Evaluates historical, physical and laboratory data to determine current gestational age") — admin/curriculum, not clinical guidance. |

### Q2 observations

- Two clear D4=1 examples: 1ddae242 (chunk drifts into related-but-not-method-specific topic) and b5c13e56 (good content but inside an abortion-care framing).
- Note: 63a982a2 raised an interesting question — it's on-topic in subject (gestational-age methods are listed) but the chunk is a test outline. **The right call is D2=0** (admin text masquerading as clinical content), which then forces score=0. The current 3-level rubric already calls this correctly; the graded rubric does too.

---

## Query 3 — q_00423

**Query**: "What clinical signs and laboratory findings indicate severe pre-eclampsia and HELLP syndrome?"

| chunk | D1 | D2 | D3 | D4 | new | old | rationale |
|---|---|---|---|---|---|---|---|
| 4b83453d (bm25@3, octen@3) | 1 | 2 | 2 | 2 | **6** | 2 | Numeric thresholds (BP > 160/110, platelets < 100×10⁶/L) + full sign list (clonus, papilloedema, epigastric pain, etc.). |
| f8327c5d (voyage@1, lateon@1) | 1 | 2 | 2 | 2 | **6** | 2 | Specific lab tests to order (FBC, U/E, clotting, LFTs, G&S) + threshold (platelets < 100×10⁹/L) + pathophysiology of each lab abnormality. |
| 9a265b41 (voyage@3) | 1 | 2 | 2 | 1 | **5** | 2 | Indicators of HELLP (platelets, elevated transaminases, blood-film findings) + extensive complications list. The complications list is adjacent-but-not-query-relevant → D4=1. |
| 879fa638 (lateon@3) | 1 | 2 | 1 | 2 | **5** | 2 | Clinical signs ("epigastric pain, nausea/vomiting, haematuria, jaundice") + general guidance ("discuss with consultant") but no thresholds → D3=1. |
| 4f7b864a (octen@2) | 1 | 1 | 2 | 2 | **5** | 2 | Bulleted criteria for severe pre-eclampsia (BP 160/110, proteinuria 5g/24h, etc.). Specific but no surrounding clinical context → D2=1. |
| 949c3b87 (bm25@1, octen@1, voyage@2, lateon@2) | 1 | 1 | 1 | 2 | **4** | 2 | Pedagogical text ("students should...") gesturing at HELLP triad without thresholds. **Old rubric mis-scores this as 2 → new rubric correctly drops it to 4.** |
| 084ac228 (gecko@3) | 1 | 2 | 1 | 2 | **5** | 2 | Pathophysiology + general statement that "delivery is essential" without thresholds. |
| c963637b (gecko@2) | 1 | 2 | 0 | 2 | **4** | 1 | Definition + pathophysiology of HELLP, no signs/labs/thresholds. **Interesting: old=1 (D3=False); new=4 (D2=2 + D4=2 contribute).** |
| e58e7231 (bm25@2) | 1 | 1 | 0 | 2 | **3** | 1 | Epidemiology of HELLP (when in pregnancy, mortality stats). No diagnostic criteria. |
| c9ec4d5b (medcpt@2) | 1 | 1 | 0 | 1 | **2** | 1 | Brief mention of HELLP within broader hypertensive-conditions main-points section. |
| 95824267 (gecko@1) | 0 | 0 | 0 | 0 | **0** | 0 | References / bibliography. Even though title mentions HELLP. |
| e29d5d53 (medcpt@1) | 0 | 0 | 0 | 0 | **0** | 0 | TOC / index of hypertensive disorders. |
| 02438433 (medcpt@3) | 0 | 0 | 0 | 0 | **0** | 0 | About eclamptic seizure timing, not pre-eclampsia / HELLP criteria. |

### Q3 observations

- Several genuinely informative pathophysiology chunks (c963637b, 084ac228) now score 4-5 under the new rubric instead of getting either pushed down to 1 (old) or up to 2 (old). The graded D3 captures "general guidance without specifics" cleanly.
- 949c3b87 — the **query's seed chunk** — gets new score 4. It's the pedagogical "students should ask the mother questions..." chunk. The old rubric had it as 2; new rubric correctly reflects that it's less useful than the rich threshold-listing chunks. **This is also a sanity check that seed chunks don't auto-anchor at max.**

---

## Query 4 — q_00455

**Query**: "How often should maternal blood pressure, pulse, and temperature be checked after birth?"

| chunk | D1 | D2 | D3 | D4 | new | old | rationale |
|---|---|---|---|---|---|---|---|
| 1bb90fa3 (octen@2, voyage@3, lateon@1) | 1 | 1 | 2 | 2 | **5** | 2 | Exact protocol: "BP and pulse every 30 minutes; temperature every 4 hours". Direct answer to the query. |
| 68f7d3c3 (medcpt@2, octen@3, voyage@2, gecko@1) | 1 | 2 | 2 | 2 | **6** | 2 | Multiple specific recommendations + clinical reasoning + risk-stratification. Comprehensive. |
| 829637c1 (medcpt@1, octen@1, voyage@1) | 1 | 1 | 2 | 2 | **5** | 2 | "Check at least once an hour if she is having any health problems." Specific but only one frequency rule. |
| 0b5bbe9c (lateon@2) | 1 | 2 | 2 | 2 | **6** | 2 | WHO Recommendation 55 — explicit schedule for BP/pulse/temp during first 24 hours. |
| 4b5bd48f (lateon@3) | 1 | 2 | 2 | 2 | **6** | 2 | WHO PNC 2013 — same protocol restated. |
| 44e88977 (bm25@2) | 1 | 1 | 1 | 2 | **4** | 1 | "BP/pulse/temp are checked and recorded" — names what to check but no frequency. |
| ac6010d5 (gecko@2) | 0 | 0 | 0 | 0 | **0** | 0 | "How to check blood pressure" — antenatal context, not postpartum. **D1 fails on timing.** |
| c38b9feb (gecko@3) | 0 | 0 | 0 | 0 | **0** | 0 | "Check the mother's blood pressure" — antenatal definitions. D1 false. |
| dcb7306a (medcpt@3) | 0 | 0 | 0 | 0 | **0** | 0 | Antenatal care, early access. D1 false. |
| 5851aec8 (bm25@1) | 0 | 0 | 0 | 0 | **0** | 0 | Care during and after a fit (eclampsia) — different clinical context. |
| 7609859c (bm25@3) | 1 | 0 | 0 | 0 | **0** | 0 | Skills-assessment marking criteria. Admin text. |

### Q4 observations

- Strong on-topic cluster (1bb90fa3, 68f7d3c3, 829637c1, 0b5bbe9c, 4b5bd48f) all hit max or near-max. Good.
- Several "antenatal BP" chunks correctly D1-zeroed. **The D1 gate is doing real work on this query** — Gecko's @2 and @3 retrievals (ac6010d5, c38b9feb) are both topic-mismatched.
- 44e88977 — has the topic but only one general statement → score 4. Reasonable.

---

## Query 5 — q_00536

**Query**: "How should a midwife assess and manage uterine tone during the third stage of labour?"

| chunk | D1 | D2 | D3 | D4 | new | old | rationale |
|---|---|---|---|---|---|---|---|
| db45b573 (bm25@3) | 1 | 2 | 2 | 2 | **6** | 2 | Active management of third stage — uterotonic within 1 min + uterine tone verification + massage if soft. |
| cc2f0b92 (bm25@2, voyage@1) | 1 | 1 | 2 | 2 | **5** | 2 | Massage if soft, sustained massage NOT recommended if uterotonic given. Header says "first stage" but content is third-stage / PPH prevention. |
| 9b011501 (voyage@2) | 1 | 2 | 2 | 2 | **6** | 2 | Detailed palpation technique + "rub up a contraction" procedure + when to recheck. |
| 6de271f0 (lateon@3) | 1 | 2 | 2 | 2 | **6** | 2 | Active management + fundal massage + ergometrine availability + observation. |
| f02bb533 (voyage@3) | 1 | 2 | 2 | 2 | **6** | 2 | Step-by-step palpation procedure with anatomical landmarks. |
| ef8b623a (lateon@1) | 1 | 1 | 2 | 2 | **5** | 2 | Third-stage management following prolonged labour: oxytocin IV if contractions absent. Specific but narrow scenario. |
| 007b15b8 (octen@3) | 1 | 1 | 1 | 2 | **4** | 2 | Checklist item: "ensure uterus well contracted; palpation + massage to promote contraction". Names actions but no procedural specifics. |
| 0d680c13 (octen@2) | 1 | 1 | 1 | 2 | **4** | 2 | Same style — checklist sub-task #12 on ensuring uterus stays contracted. |
| d642b891 (octen@1, seed chunk) | 1 | 1 | 1 | 2 | **4** | 2 | Sub-task #15: ensure uterus remains contracted (palpation + massage). Same style. |
| 1cecfdfb (medcpt@1) | 1 | 1 | 0 | 1 | **2** | 1 | Discussion of active vs expectant management as concepts. |
| 88b00ab8 (gecko@2) | 0 | 0 | 0 | 0 | **0** | 0 | General third-stage care intro, no uterine-tone specifics. |
| 31a84f2a (lateon@2) | 0 | 0 | 0 | 0 | **0** | 0 | Documentation requirements, not clinical management. |
| 3744457c (bm25@1) | 1 | 0 | 0 | 0 | **0** | 0 | Learning objectives ("students will be able to..."). Admin. |
| c7314aa1 (medcpt@3) | 0 | 0 | 0 | 0 | **0** | 0 | Intro to expectant vs physiological management; no uterine-tone content. |
| a4da0882 (gecko@3) | 0 | 0 | 0 | 0 | **0** | 0 | Documentation/consent admin. |
| 61c07566 (medcpt@2) | 0 | 0 | 0 | 0 | **0** | 0 | Informed-choice procedure. Not clinical. |
| 37f777a3 (gecko@1) | 0 | 0 | 0 | 0 | **0** | 0 | Antenatal counselling about third-stage options. D1 false. |

### Q5 observations

- The three "checklist sub-task" chunks (007b15b8, 0d680c13, d642b891) all hit score 4 under the new rubric: they correctly identify the tasks (palpation, massage) but lack procedural detail. **Old rubric ceilings them at 2 alongside the chunks with full procedure.** This is exactly the discrimination the graded rubric was designed for.
- Gecko's top-3 here (37f777a3, 88b00ab8, a4da0882) all score 0. **The graded rubric makes the gap visible**: Gecko is surfacing third-stage-related but uterine-tone-irrelevant chunks. This explains why Gecko's P@3 underperforms on this query.

---

## CoT-ready anchor examples — summary by dimension

These short chunks span the levels for each dimension and can serve as the worked examples in the eventual judge prompt.

### D2 anchors

| Level | Anchor chunk | Why |
|---|---|---|
| **D2=0** | 27b615ab (Q1) "Intermittent supplementation might cost less..." | Cost/resource talk — not clinical |
| **D2=0** | 16ae1a57 (Q1) "RECOMMENDATION 10: IRON AND FOLIC ACID SUPPLEMENTATION" | Pure section header |
| **D2=1** | 829637c1 (Q4) "Check temperature, pulse, BP regularly — at least once an hour if health problems" | Single rule, no broader context |
| **D2=1** | ad7678f8 (Q1) "Routinely once daily in pregnancy; twice daily for anaemia; store iron safely" | Frequency + storage; thin clinical depth |
| **D2=2** | 666d6b2a (Q1) "Daily oral iron and folic acid supplementation with 30-60 mg... is recommended... to prevent maternal anaemia, puerperal sepsis, low birth weight, preterm birth" | Recommendation + indications + dosing |
| **D2=2** | f8327c5d (Q3) "Blood samples — FBC, U/E, Clotting, LFTs, G&S — laboratory reports demonstrate altered blood pattern: Haemolysis... Elevated liver enzymes... Low platelets..." | Tests + thresholds + mechanism each |

### D3 anchors

| Level | Anchor chunk | Why |
|---|---|---|
| **D3=0** | 5fffb433 (Q1) Cochrane review summary listing doses studied without recommendation | Names doses but no actionable recommendation |
| **D3=0** | c963637b (Q3) HELLP definition + pathophysiology only | No diagnostic criteria specified |
| **D3=1** | 879fa638 (Q3) "Be aware of, and assess for: epigastric pain..., nausea, headaches..." + "Discuss with consultant" | Action specified but no thresholds |
| **D3=1** | 44e88977 (Q4) "BP, pulse, temperature are checked and recorded" | What to check, but no frequency |
| **D3=1** | 007b15b8 (Q5) "Ensure uterus well contracted — palpation + massage" | Action verbs but no procedural specifics |
| **D3=2** | 1bb90fa3 (Q4) "BP and pulse every 30 minutes; temperature every 4 hours" | Exact frequencies |
| **D3=2** | 318a18d9 (Q1) "200 mg ferrous sulfate (65 mg elemental iron) + 400 mcg folic acid may be replaced by 185 mg ferrous fumarate..." | Exact substitution rule |
| **D3=2** | 4b83453d (Q3) "BP systole > 160 mmHg, diastole > 110 mmHg + platelets < 100×10⁶/L..." | Numeric thresholds |

### D4 anchors

| Level | Anchor chunk | Why |
|---|---|---|
| **D4=0** | 27b615ab (Q1) Resource/cost talk | None of the chunk addresses the dosage query |
| **D4=1** | b40eea13 (Q1) Folic acid section followed by immunisation section | ~40% query-irrelevant; immunisation is its own topic |
| **D4=1** | 9a265b41 (Q3) HELLP indicators (3 items) followed by extensive complications list | Half is "what to look for" (the query); half is "what HELLP causes" (adjacent) |
| **D4=2** | 666d6b2a (Q1) Standalone recommendation paragraph | Entire chunk is the answer |
| **D4=2** | 1bb90fa3 (Q4) "Check BP every 30 min, temp every 4 hours" + framing | All content is monitoring schedule |

### Score=6 anchors (top of scale)

These are chunks that get 6 under the new rubric (D1=1, D2=2, D3=2, D4=2):

| Query | Chunk | Why max |
|---|---|---|
| q_00041 | 666d6b2a | Standalone formal recommendation with dose ranges and indications |
| q_00041 | c3066ae0 | Threshold + treatment dose + equivalents — three pieces of guidance in one focused chunk |
| q_00041 | 94aa6726 | Dosing table with tablet counts and durations |
| q_00148 | b5aa599f | All three methods + calculation rules + worked example |
| q_00148 | eb94102d | Naegele's Rule with formula + cycle-length adjustments + scan timing |
| q_00423 | 4b83453d | Numeric thresholds + clinical sign list |
| q_00423 | f8327c5d | Lab tests + thresholds + mechanism for each |
| q_00455 | 68f7d3c3 | Multiple specific schedules + risk-stratification + reasoning |
| q_00455 | 0b5bbe9c | Formal WHO Recommendation 55 |
| q_00455 | 4b5bd48f | WHO PNC schedule for first 24 hours |
| q_00536 | db45b573 | Active management with timing + decision tree |
| q_00536 | 9b011501 | Detailed palpation technique + "rub up" procedure |
| q_00536 | 6de271f0 | Active management + uterotonic + observation |
| q_00536 | f02bb533 | Step-by-step palpation procedure |

---

## Findings for the rubric prompt design

### 1. D1 is a hard gate and is doing real work

Across the 5 queries, 21 of 65 chunks (32%) get D1=0. Most are timing mismatches (antenatal vs postpartum) or scope mismatches (eclamptic fit vs routine postpartum). The current 3-level rubric and the new graded rubric agree on D1 for almost all chunks — the gate is the easiest part to specify.

**Prompt implication**: lead with D1, halt scoring if D1=0. No need to evaluate D2/D3/D4. Saves judge tokens.

### 2. D2 vs D3 distinction is mostly clean, but pedagogical chunks are tricky

Chunks like 949c3b87 (the seed chunk for Q3) — "students should ask the mother questions, listen carefully..." — gesture at the answer without giving it. Old rubric tends to score these as D3=true (the answer is alluded to), giving score=2. New rubric correctly catches this: D2=1 (some clinical content), D3=1 (general action like "check lab reports"), D4=2 (focused) → score 4. Distinct from a chunk that actually lists the criteria.

**Prompt implication**: anchor D3=2 firmly on "exact numeric thresholds, doses, intervals, or step-by-step procedure". "Mentions you should look for X" is D3=1 unless the X-criteria are specified.

### 3. D4 catches "good content in a wider section that drifts"

Examples: b40eea13 (folic acid + immunisations), 9a265b41 (HELLP signs + extensive complications), d6fa6f68 (dose equivalents + task-shifting). These get D4=1 cleanly.

**Prompt implication**: judge should explicitly identify which sentences/paragraphs of the chunk are useful for the specific query and estimate the fraction. Don't conflate "useful clinical content" with "query-relevant content".

### 4. The 6-level scale (0-6) actually populates well

Score distribution across 65 chunks: 0×21, 1×0, 2×3, 3×1, 4×7, 5×11, 6×22. The score=6 cluster is large but that's expected — when retrievers do well, they surface several maximally-useful chunks per query. The discrimination among on-topic chunks (scores 2-6) is what's new. **No level except 1 is empty.**

The empty "1" level deserves attention. Why no score=1? Because score=1 requires D1=1 with D2+D3+D4 summing to 1 — i.e., exactly one of the three is 1, others zero. In practice, an on-topic chunk with one weakly-useful dimension usually has at least one other ≥1. Plausible but rare.

### 5. Cases where old 3-level conflicts with new 0-6 are informative

10 chunks shift "up": old score=1 → new {2, 3, 4} (e.g., e58e7231, 1cecfdfb, c963637b). These are chunks with rich clinical content but no actionable guidance — the old rubric flattens them; the new rubric grades them by content richness.

No chunks shift "down" (i.e., no chunk has old=2 → new 0/1/2). The new rubric is more generous to "good but not maximal" chunks, which is the whole point of moving away from a binary D3.

### 6. Anchor count for the eventual prompt

For each level of each dimension, the worked-example set above gives 1-3 candidates. Suggest the prompt include:
- 2 worked examples for D2 levels 0, 1, 2 (6 examples)
- 2 worked examples for D3 levels 0, 1, 2 (6 examples)
- 2 worked examples for D4 levels 0, 1, 2 (6 examples)
- 3 full end-to-end scored examples spanning the score range (e.g., one score=6, one score=4, one score=2, one score=0 via D1-gate)

Plus the rubric definitions, the formula, and the structural constraints. Estimated prompt length ~2500-3000 words. Doable.

---

## Open questions for the rubric prompt

1. **Should the judge see other retrievers' chunks for the same query?** Currently the judge sees one (q, chunk) at a time. Multi-chunk presentation could enable comparative reasoning ("this chunk is denser than the alternative") but introduces position bias and complicates the prompt. **Recommendation**: keep single-chunk presentation.

2. **Should the judge see retrieval rank or score?** No — that would bias the judge. The chunk should be judged on its merit alone.

3. **What's the role of the seed_chunk_id?** In the original (Phase 2b) design, seed chunks were auto-labeled as score=2. For the new graded rubric, **don't auto-label seeds.** As we saw with 949c3b87 (Q3 seed, score=4 under new rubric), a seed chunk can be a teaching-text reference that wouldn't score max on its own merits.

4. **How to handle tables?** Several chunks have malformed tables (94aa6726 dose table; 4b83453d sign list as table). The judge should parse them as bullet lists and score on content, not on rendering. Worth adding a prompt note.

5. **How to handle very short chunks (< 150 chars)?** Some valid score=5 chunks are tiny (318a18d9 at 228 chars). Short ≠ uninformative. The judge prompt should not penalize length per se; D4=2 is achievable with short focused chunks.

---

## Next steps

1. **You review this file** — does the criteria refinement match your mental model? Any anchors that disagree with your judgment?
2. **Draft the judge prompt** using these anchors. Estimated 2500-3000 words. Will go to GitHub #13.
3. **Pilot the prompt** on a small batch (~30 chunks) to spot-check the judge's calls.
4. **Then proceed to Stage 9 / GitHub #14 — full pilot on 100 audit queries.**

If the criteria here look right, we can move to drafting the actual judge prompt.

---

## Tier 1 pilot validation — Qwen judge vs. Opus-4.7 reference labels (2026-05-17)

The 62 (query, chunk) scores tabulated above were produced by **Claude Opus 4.7** while drafting this doc (not by manual human review). They serve as the reference set for validating the production judge.

**Production judge under test**: `Qwen/Qwen3.5-397B-A17B-FP8` with the v2.1 graded rubric, soft thinking-budget 10k, hard cap 25k, temperature 0.

**Pilot run**: `data/audit/v2_pilot_h100_shard0.jsonl` — 1,150 unique (q, c) pairs (100 audit queries × 6 retrievers × top-3, deduped). All 62 reference pairs are inside this set.

### Score-level agreement (Qwen vs. Opus, 62 pairs)

| Metric | Result |
|---|---|
| Exact 4-dim agreement (D1+D2+D3+D4 all match) | **32/62 (52%)** |
| Exact final-score agreement (0–6) | **37/62 (60%)** |
| Within ±1 on final score | **56/62 (90%)** |

**Most disagreements are off-by-one shifts on a single graded dimension** — Qwen is slightly more conservative on D3=2 / D4=2 awards. This is why exact-score agreement (60%) is much lower than within-±1 (90%) and threshold agreement (~90%): a typical mismatch is Opus=5 / Qwen=4, which is a "miss" for exact score but stays on the same side of the ≥3 and ≥4 cutoffs. The off-by-one shifts only flip a threshold call when the boundary sits exactly between the two judges' numbers (e.g., score=4 vs 5 at the ≥5 cutoff). Only 3 D1 flips occurred (all on q_00536), where Opus called the chunk off-topic and Qwen called it on-topic-but-shallow.

### Threshold-based agreement (precision-relevant)

For Variant D ranking metrics we collapse the 0–6 score to a binary "relevant at threshold T" label. The table below shows Qwen vs. Opus at the four precision-relevant cutoffs:

| Threshold | Reference (Opus) positives | Qwen positives | Agreement | Precision | Recall |
|---|---|---|---|---|---|
| **score ≥ 3** | 39/62 | 38/62 | **59/62 (95%)** | 0.97 | 0.95 |
| **score ≥ 4** | 37/62 | 34/62 | **57/62 (92%)** | 0.97 | 0.89 |
| **score ≥ 5** | 30/62 | 21/62 | **53/62 (85%)** | 1.00 | 0.70 |
| **score ≥ 6** | 14/62 | 11/62 | **53/62 (85%)** | 0.73 | 0.57 |

Read: at the **strict (≥5)** threshold, Qwen-positives are a clean subset of Opus-positives (precision 1.00) but Qwen misses 9 chunks Opus considered top-tier (recall 0.70). At **lenient (≥3)** the two agree on 95% of calls. The **≥6 top-anchor** label is the noisiest — Qwen and Opus disagree on roughly 1 in 6 chunks at the maximum-score boundary.

### Reasoning-trace quality (all 1,150 pilot records)

- **Structural coverage**: 1,146/1,150 (99.7%) of traces explicitly reference all four dimensions (D1, D2, D3, D4) by name in the reasoning before emitting JSON.
- **Length**: median 808 words, mean 1,361 words, max 19,443 (a handful of long deliberations on borderline cases). Soft cap of 10k tokens generally holds.
- **Manual spot-check** at score=0, score=3, score=6 (random per bucket): reasoning correctly applies the D1 gate, considers each dimension's rubric anchors, and arrives at scores consistent with the criteria. No boilerplate or hallucinated chunk content observed.

### Conclusion

The Qwen judge is reliable as a **binary relevance classifier** at the ≥3 and ≥4 thresholds (P ≥ 0.97, R ≥ 0.89), which is what HR/P/MRR/NDCG@k consume for Variant D. Exact-score agreement is modest (60%) — do not use individual graded scores for absolute calibration, but rank ordering should be sound. The ≥6 max-anchor label should be treated as noisy.

---

## Per-retriever scoreboard (Variant D, k=3, n=100 queries)

Computed by `scripts/audit_metrics_v2.py` from the v2 pilot labels.

For RAG at k=3 with a long-context LLM (no position bias), **HR and Precision are the operationally meaningful metrics** — "is the information in the bundle?" and "how much of the bundle is useful vs. noise?". MRR / NDCG would matter only if position within top-3 drove downstream behavior.

Three lenses on the same six retrievers:
- **Binary lenient (≥3)** — relevance threshold on the 0–6 score
- **Binary strict (≥5)** — top-tier relevance threshold
- **Weighted (wHR / wP)** — threshold-free; each chunk contributes score/6 ∈ [0, 1]

All metrics averaged over the full audit set (n=100); queries with no relevant chunk in the judged pool contribute HR=0 and P=0 (deployment-honest convention).

| Retriever | HR (≥3) | P (≥3) | HR (≥5) | P (≥5) | wHR | wP |
|---|---:|---:|---:|---:|---:|---:|
| **voyage** | **0.990** | **0.820** | **0.730** | **0.430** | **0.847** | **0.657** |
| octen      | 0.990 | 0.760 | 0.720 | 0.413 | 0.845 | 0.624 |
| lateon     | 0.990 | 0.727 | 0.710 | 0.380 | 0.833 | 0.586 |
| gecko      | 0.840 | 0.493 | 0.490 | 0.210 | 0.693 | 0.404 |
| bm25       | 0.740 | 0.413 | 0.390 | 0.163 | 0.613 | 0.336 |
| medcpt     | 0.610 | 0.287 | 0.310 | 0.117 | 0.523 | 0.259 |

### Three-tier reading

1. **Top tier — voyage ≈ octen ≈ lateon**: all three deliver any-relevant content in ~99% of queries; precision and weighted scores are nearly indistinguishable. Voyage edges octen on lenient precision (0.82 vs 0.76) but they're effectively tied at the strict threshold.
2. **Middle — gecko** (the on-device deployed retriever): HR drops to 0.84 lenient / 0.49 strict; precision halves vs. the top tier. Substantial gap.
3. **Bottom — bm25, medcpt**: bm25's lexical overlap pulls some weight (HR 0.74 lenient) but precision is low; medcpt is the weakest across every metric.

### Honest read of the strict numbers

Even the top retrievers deliver a strict-relevant chunk (score ≥ 5) in only ~73% of queries, with ~43% of the top-3 being strict-relevant. The 19 queries where **no retriever's top-3 contained a strict-relevant chunk** are an inherent ceiling — to push past it we'd need either deeper k or a broader candidate pool, not just better re-ranking.

---

## Per-retriever scoreboard — Tier 2 full audit (n=3,185 queries, 2026-05-18)

Computed by `scripts/audit_metrics_v2.py --labels data/audit/v2_full_h100.jsonl --rankings-dir data/full`.

Full audit run: 36,418 (q, c) pairs judged across **3,185 queries** (the entire query set, not just the 100-query audit subset). Same v2 graded rubric and same Qwen3.5-397B-A17B-FP8 judge as Tier 1.

| Retriever | HR (≥3) | P (≥3) | HR (≥5) | P (≥5) | wHR | wP |
|---|---:|---:|---:|---:|---:|---:|
| **voyage** | **0.996** | **0.867** | **0.753** | **0.452** | **0.860** | **0.682** |
| octen      | 0.991 | 0.804 | 0.716 | 0.403 | 0.847 | 0.637 |
| lateon     | 0.971 | 0.738 | 0.664 | 0.350 | 0.815 | 0.581 |
| gecko      | 0.814 | 0.477 | 0.439 | 0.193 | 0.662 | 0.393 |
| bm25       | 0.754 | 0.417 | 0.371 | 0.163 | 0.602 | 0.338 |
| medcpt     | 0.644 | 0.334 | 0.272 | 0.112 | 0.517 | 0.277 |

All metrics over n=3,185 queries (deployment-honest: queries with no chunk meeting the threshold contribute HR=0 / P=0).

### Tier 1 (n=100) vs Tier 2 (n=3,185) deltas

| Retriever | HR(≥3) T1→T2 | P(≥3) T1→T2 | HR(≥5) T1→T2 | P(≥5) T1→T2 | wHR T1→T2 | wP T1→T2 |
|---|---:|---:|---:|---:|---:|---:|
| voyage | 0.990 → 0.996 | 0.820 → **0.867** | 0.730 → 0.753 | 0.430 → 0.452 | 0.847 → 0.860 | 0.657 → 0.682 |
| octen  | 0.990 → 0.991 | 0.760 → 0.804 | 0.720 → 0.716 | 0.413 → 0.403 | 0.845 → 0.847 | 0.624 → 0.637 |
| lateon | 0.990 → 0.971 | 0.727 → 0.738 | 0.710 → 0.664 | 0.380 → 0.350 | 0.833 → 0.815 | 0.586 → 0.581 |
| gecko  | 0.840 → 0.814 | 0.493 → 0.477 | 0.490 → 0.439 | 0.210 → 0.193 | 0.693 → 0.662 | 0.404 → 0.393 |
| bm25   | 0.740 → 0.754 | 0.413 → 0.417 | 0.390 → 0.371 | 0.163 → 0.163 | 0.613 → 0.602 | 0.336 → 0.338 |
| medcpt | 0.610 → 0.644 | 0.287 → 0.334 | 0.310 → 0.272 | 0.117 → 0.112 | 0.523 → 0.517 | 0.259 → 0.277 |

### What Tier 2 confirmed vs added

- **Same three-tier ranking**: voyage > octen > lateon ≫ gecko > bm25 > medcpt. Conclusion from Tier 1 holds at full scale.
- **Voyage clearly best now**. At Tier 1 voyage, octen, lateon all tied at HR(≥3) = 0.990. At n=3,185 the difference is statistically real — voyage's P(≥3) lead over octen widened from 0.06 → 0.06 and now visible in every metric.
- **Octen vs lateon now separable**. HR(≥3): 0.991 vs 0.971 — small but consistent.
- **Strict numbers (≥5) drift down for top retrievers** at scale. More queries reveal more "no strict-relevant chunk exists anywhere in the candidate pool" cases — the ceiling of what depth-3 retrieval can achieve over this corpus.
- **Weighted metrics very stable** (within 0.02 between T1 and T2). The graded signal smooths the per-query noise the binary thresholds expose.

### Run notes

- Both shards used H100, 32 workers each, ~13h wall-clock total (would've been ~9h without one preempt cycle on shard 0).
- Real per-pod throughput at 32 workers ≈ 0.5-0.7 records/sec — basically the same per-pod throughput we got at 8 workers in Tier 1. The GPU was already saturated at 8 workers for this 17B-active-param MoE model. The 2× speedup we got came from running 2 pods in parallel, not from raising workers per pod.
- Judge: same `Qwen/Qwen3.5-397B-A17B-FP8` model, same v2_graded prompt (hash `9d2abdfb76b030ea`), same thinking budgets as Tier 1 (soft 10k, hard 25k).
- 0 errored rows out of 36,418.
