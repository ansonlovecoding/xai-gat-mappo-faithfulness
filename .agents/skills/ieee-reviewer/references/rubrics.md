# Scoring rubrics

## Answer grades (per question, 0–5)

| Grade | Anchor | 中文描述 |
|---|---|---|
| 5 | Precise, evidence-backed, cites the artefact or number, and states the limitation unprompted | 精确、有证据、主动交代局限 |
| 4 | Correct and specific; minor imprecision or missed nuance | 正确具体，略有不精确 |
| 3 | Directionally right but vague — the idea is there, the evidence is not | 方向对但含糊，拿不出证据 |
| 2 | Partially wrong, or "I don't know" WITH a sensible plan to find out | 部分错误；或"不知道"但说得出如何查证 |
| 1 | Evasive, bluffing, or "I don't know" with no plan | 回避、虚张声势、纯粹不知道 |
| 0 | Contradicts their own artefacts, or refuses to engage | 与自己的数据/文档矛盾 |

Bluffing detection: if the answer asserts a number or fact you can
check, check it. A confident wrong number grades 0–1 and the report
notes it — real reviewers remember exactly these moments.

## Dimension scores (final report, each 1–10)

IEEE-conference anchors: 9–10 = best-paper candidate; 7–8 = clear
accept material on this dimension; 5–6 = borderline, needs revision;
3–4 = major weakness, likely reject driver; 1–2 = fatally flawed.

- **Novelty** — is the *problem framing* (faithfulness decoupling,
  AoI-grounded degradation) genuinely new, and is the delta over
  "attention is not explanation" literature articulated?
- **Technical soundness** — metric definitions coherent (DEF, margin
  variant, WAMSN), mechanism correctly implemented (freeze semantics),
  no logical gaps between claim and evidence.
- **Experimental rigor** — seeds, held-out demand, paired designs,
  statistical tests appropriate (clustering!), baselines fair,
  negative results reported.
- **Clarity & presentation** — three-act story lands, figures readable,
  terms defined before use, deviations documented.
- **Reproducibility** — pinned versions, manifests, committed
  checkpoints, one-command reruns; penalize anything that exists only
  on one machine.

## Overall score and decision

Overall /10 = weighted mean: soundness 30 %, rigor 25 %, novelty 20 %,
clarity 15 %, reproducibility 10 %. **Adjust by Q&A performance**: mean
answer grade ≥4 → +0.5 (the authors clearly command their material);
mean ≤2 → −0.5. Cap to [1, 10].

| Overall | Decision analog |
|---|---|
| ≥ 7.5 | Accept |
| 6.0–7.4 | Minor revision |
| 4.5–5.9 | Major revision |
| < 4.5 | Reject & resubmit |

State the decision with one honest paragraph on what most helped and
most hurt the score.
