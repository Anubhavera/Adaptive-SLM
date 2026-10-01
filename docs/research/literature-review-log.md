# Literature review scope — 2026-10-01

This is a bounded prior-art search, not a systematic review or a certification of novelty. Publications are linked in the [proposal](research-direction.md) and [project audit](2026-10-01-audit-and-plan.md). Preprints, project pages and peer-reviewed proceedings are not interchangeable publication statuses; no venue acceptance is inferred from search snippets.

The search covered compact mobile models, local tool calling, Android agents, state/plan validity, execution assurance, interruption recovery, resource-aware replanning, context management, memory and sustained mobile inference. Representative final-pass queries included:

- `on device mobile agent interruption recovery checkpoint idempotent actions resource budget small language model`
- `mobile agent persistent execution recovery memory budget replanning phone on-device research`
- `site:arxiv.org on-device agents state verification recovery energy adaptive`
- `site:arxiv.org mobile assistant transactional execution duplicate actions interruption`
- `site:arxiv.org "on-device" "agent" "process" "recovery"`
- `site:arxiv.org "mobile" "agent" "cold" "restart"`
- `site:arxiv.org "Android" "agent" "lifecycle"`
- `site:arxiv.org "agent" "model unloading"`

Search result absence is not evidence of absence. Older “mobile agent” migration/checkpointing literature must be distinguished from current phone-hosted language agents, but foundational fault-tolerance techniques still require attribution.

## Reading depth

Selected method, setup or limitation sections were inspected in the primary HTML versions of Extending FunctionGemma (2609.25373v1), AndroidReality (2608.07775v1), PlanFence (2609.03340v2), ADF-EA (2609.30691v1), PalmClaw (2607.13027v2) and Efficient On-Device Agents via Adaptive Context Management (2511.03728v1). Selected Agentic Plan Caching sections were also inspected. This does not mean every appendix, artifact or numerical claim has been independently reproduced.

BRACE was inspected through the authors' project page; its full paper must be acquired and reviewed before final baseline implementation. CORE was checked through the primary MLSys proceedings entry. Google FunctionGemma/Mobile Actions and Android platform/distribution constraints were checked through official documentation. Budget-Curated Memory and the mobile memory-architecture benchmark were identified through primary abstracts and need full-paper review.

Other architecture/distillation/caching papers in the audit provide a reading queue and context, not a completed systematic comparison. No cited mobile speed number is used as a measurement of our device.

## What changed after the wider search

The broad idea “budget-aware verified recovery” overlaps with ADF-EA's persistent execution semantics and BRACE's replanning controller. It must not be sold as unexplored. The proposal was narrowed to fully local Android lifecycle/model-residency costs, with these existing methods and their combination required as baselines. Even that narrower distinction remains unverified until citation tracing and a pilot demonstrate a meaningful weakness of existing approaches.

## Next review gate

For each closest source, record its assumptions, released code/license, inference location, hardware, interruption model, evidence guarantees, baseline strength and limitations. Trace backward references and forward citations, and search newer versions again before submission. State clearly which comparisons use original implementations and which are our adaptations. If a stronger prior method already solves the pilot cheaply, revise the research question before major training.
