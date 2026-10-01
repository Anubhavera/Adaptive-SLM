# Proposed paper direction: lifecycle-aware recovery for on-device assistants

Date: 2026-10-01. This is a research proposal and a bounded literature review, not a novelty certification or a prediction of acceptance. Selected methods, experimental setup and limitations of the closest papers were inspected; additional citation tracing and full-paper review are needed before committing to a novelty claim.

## Recommended question

**Can a sub-billion-parameter, fully local Android assistant recover efficiently after process death or model unloading, while preserving verified progress when user data changes?**

Working title: **Lifecycle-Aware Recovery for Fully Local Android Assistants**.

The wider search found substantial overlap with the initial budget-aware recovery idea. ADF-EA already connects persistent execution evidence with authorized recovery, and BRACE already controls replanning cost. The narrower lifecycle question below is a candidate to investigate, not an established new contribution. Porting those methods to a phone, or combining them, does not alone establish algorithmic novelty.

The product vision remains a persistent personal assistant that chats, uses permitted phone capabilities and helps autonomously. The first paper needs a narrower claim than “Apple Intelligence on every phone.” Stock Android with user-granted permissions is confirmed. The M35 is the first device; the lowest supported tier is still to be identified. Notes, task list and calculator provide controlled initial tasks, not the final product boundary.

The proposed contribution is an execution/recovery method and evaluation protocol. It is not a claim of a new foundation model. Standard database journaling, idempotency, schema validation, permissions and checkpointing are engineering components, not independently novel inventions.

## Closest work and overlap we must acknowledge

| Work | Established direction | Implication for this project |
|---|---|---|
| [FunctionGemma Mobile Actions](https://ai.google.dev/gemma/docs/mobile-actions) | Existing official recipe for local natural-language-to-phone-tool adaptation and deployment. | Do not claim the first small offline phone assistant. Reuse as a baseline. |
| [Extending FunctionGemma, September 2026](https://arxiv.org/html/2609.25373v1) | A 270M model extended to fifteen action categories. The reported evaluation is mainly single-turn exact match. The device demo uses a Pixel 8 Pro with 12 GB RAM; its quantized artifact differs from the weights used for accuracy scoring. | Broader tool coverage alone overlaps. Compare multi-step interrupted execution on the actual quantized mobile artifact. Its limitations explicitly identify multi-turn interaction and cross-architecture evaluation as future work. |
| [TinyAgent](https://arxiv.org/abs/2409.00608) and [AutoDroid-V2](https://arxiv.org/abs/2412.18116) | Small-model tool orchestration, retrieval, app documentation and script-based execution. | Neither tool retrieval nor generating action programs is a sufficient novelty claim. |
| [AndroidReality](https://arxiv.org/html/2608.07775v1) | Controlled UI/action/transition perturbations and test-time recovery, including restarts and permission dialogs. Its evaluated 2B–32B agents are served on A100 GPUs with a separate emulator. | “Mobile agents should recover” is already explored. A potential distinction is mobile-local sub-billion inference, persistent execution state and recovery-cost decisions measured on stock phones. This distinction is an inference from the inspected setup, not proof of an untouched field. |
| [PlanFence / Fresh Memory, Stale Plans](https://arxiv.org/html/2609.03340v2) | Dependency-scoped validation of the inputs used to derive an action; fresh observations alone do not make an old plan valid. Its guarantee is conditional and does not make checking and external execution atomic. | Use a dependency-validation baseline. Do not present stale-plan checking as new or claim exactly-once effects across arbitrary third-party apps. |
| [ADF-EA, September 2026](https://arxiv.org/html/2609.30691v1) | Shared capability contracts, persistent progress/uncertainty and evidence-governed retries or repair. Its inspected evaluation uses simulated device domains; real-device deployment is an explicit future direction. | Strong overlap with our proposed journal and verification semantics. Treat these as prior foundations. A real Android implementation needs evidence about lifecycle/resource costs and an additional mechanism or a substantive empirical contribution. |
| [BRACE: Budgeted Replanning](https://nebulis-lab.com/BRACE/) | The authors' project describes admission, replanning-mode selection, token/latency budgets and composable compression. Examples cover simulated embodied environments and physical robots. | Cost-aware replanning is already explored. Adapt its approach as a baseline, and compare against its combination with execution assurance. The project description was inspected; full-paper review is still required. |
| [Efficient On-Device Agents via Adaptive Context Management](https://arxiv.org/html/2511.03728v1) | Structured context-state summaries, compact tool schemas and just-in-time schema loading, evaluated with a 3B model and qualitative quantized device tests. | A compact state summary or tool-schema retrieval is not a new contribution. Compare lifecycle recovery against compressed-state reconstruction. |
| [PalmClaw](https://arxiv.org/html/2607.13027v2) | Mobile-native agent loop, session/memory management and structured device tools. The inspected version explicitly uses remote LLM inference. | Native orchestration already exists. Our fully local inference requirement differs from this evaluated implementation, but is not by itself a new research field. |
| [Budget-Curated Memory](https://arxiv.org/abs/2606.25115) and [mobile-agent memory benchmark](https://researchportal.hkust.edu.hk/en/publications/benchmarking-memory-architectures-for-mobile-agents-in-short-term/) | Existing work studies memory utility/cost and mobile-agent memory architectures. Only abstracts were inspected in this pass. | Add to the full-reading queue before claiming budget-aware persistent memory is unexplored. |
| [Agentic Plan Caching](https://arxiv.org/abs/2506.14852) | Plan-template reuse and adaptation across similar tasks. | Workflow reuse alone overlaps. Include cache/replay comparisons only as controlled baselines. |
| [CORE, MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/136b9a13861308c8948cd308ccd02658-Abstract-Conference.html) | Joint energy-aware CPU/GPU/memory frequency scheduling for mobile LLMs. | A temperature heuristic or DVFS controller alone is insufficient. Our stock app should optimize execution/recovery decisions without assuming privileged frequency controls. |
| [MobileExplorer](https://arxiv.org/abs/2605.26546) | On-device exploration memory and rollback for GUI agents. | Memory-assisted recovery/latency gains need a comparison to this line of work; parallel exploratory interactions are not automatically appropriate for user data. |
| [Proactive triggering study](https://arxiv.org/abs/2605.30152) | A small graph controller decides when to invoke a language agent. | Cheap triggering is already explored. Keep it supporting infrastructure rather than the initial novelty claim. |

The candidate distinction is the **Android lifecycle and inference-residency decision problem**: after partial execution, the app may lose its process and in-memory context; the model may also have been unloaded to leave RAM available to the user. Recovery must reconcile mutable user state and decide whether a verified continuation suffices or a cold model load and context reconstruction are necessary. The inspected sources do not establish that exact problem as their evaluated contribution; this is an inference from a bounded review, not proof it is untouched. The obvious combination of ADF-EA, BRACE and structured-context methods is a required strong comparator.

## A concrete example

Request: “Create a shopping note and a task reminding me to buy those items.” The assistant creates the note, then its process is terminated before it records the acknowledgement. Meanwhile, the user edits the note or completes the task manually.

On restart, replaying the original sequence might create duplicates or use stale contents. Replanning everything may be correct but repeatedly loads the model and regenerates context. A recovery controller should first determine what can be observed reliably, then choose an eligible continuation or fresh plan. If the external outcome cannot be established, it should report uncertainty rather than silently repeating an effect.

For the demo's own tools, transactions and operation IDs can make effects verifiable and deduplicated. For third-party apps without a compatible contract, those guarantees are weaker. This boundary must be reported rather than hidden behind an “autonomous” label.

## Proposed method

The local model proposes typed tool actions. A deterministic executor maintains a compact persistent journal: task identity, plan version, action arguments, relevant state versions, permission state, operation identity, pending/confirmed outcome and verification evidence. Separate model context from tool-result and UI text so observed content cannot grant new instructions or permissions. The [2026 accessibility prompt-injection study](https://arxiv.org/abs/2608.08939) motivates adversarial test cases; it does not prove this design secure.

At task boundaries or recovery points, the policy chooses among:

1. Verify relevant outcomes/preconditions and resume an eligible continuation.
2. Replan from refreshed state with the local SLM.
3. Defer until the resource or permission condition permits execution.
4. Request clarification when intent or external outcome is ambiguous.

Estimate the cost of verification, model load, prefill, decoding and execution from local measurements. Start with a transparent rules-based cost policy. Consider a small learned selector only if it beats those rules on held-out tasks/devices. Resource estimates must never bypass permission checks or required correctness evidence. If safety eligibility removes all autonomous choices, the controller cannot manufacture eligibility to meet a latency deadline.

Make model residency and Android process generation explicit policy inputs. Reconstruct reasoning context from the still-valid dependency frontier and compact tool records, rather than replaying the entire conversation. These components have prior analogues; the experiment must establish an additional benefit of accounting for lifecycle/residency. Test warm continuation, deliberate model unloading and actual process restart separately. Persistent reasoning summaries are advisory; authoritative outcomes come from tool contracts and fresh observations.

A proposed objective is to minimize recovery latency/inference work subject to a specified verified-completion floor and bounded memory. Measure energy only if instrumentation supports it. Deferred, blocked and clarification-required tasks must be counted separately; refusing everything is not high task success.

## Implementation suited to the available budget

Use a Kotlin Android app, native local inference, SQLite/Room task/journal storage and explicit tool adapters. Keep assistant activation/event handling light; load the large model for active reasoning and release resources when idle. An optional default-assistant service is a product integration route, not unrestricted access. [Android VoiceInteractionService](https://developer.android.com/reference/android/service/voice/VoiceInteractionService).

Compare a compact general model such as LFM2.5 with a task specialist such as FunctionGemma, subject to access and runtime support. FunctionGemma has a distinct function-call template and is intended for task adaptation, not unrestricted direct chat. The current SmolLM2 starter does not implement its native tool protocol. A separate properly templated adapter/data path is required.

The M35 already ran the SmolLM2 pipeline-test artifact with about 15.9 generated tokens/s including prompt processing, and 385.5 MiB peak native process RSS over three short generations. This demonstrates execution viability, not agent success, sustained performance or whole-app memory. Use the [raw record](phone-native-smoke.txt) and [hardware profile](phone-profile.json).

Train one compact LoRA baseline on new multi-step/clarification examples using free sessions and portable checkpoints. Avoid training from scratch, online teachers during every batch, or large-scale reinforcement learning for this initial paper. First evaluate the system with fixed model weights so benefits cannot be attributed to changing both model and controller at once.

## Experiments that could support a paper

Create task templates with machine-checkable final state and separate user phrasing. Start with about 40–60 distinct multi-step workflows spanning controlled notes/tasks/calculation and a small set of permission-mediated Android tools. This is a proposed scale, to revise after a pilot. Hold out task compositions and argument patterns, not merely random paraphrases.

Pair each task with clean execution and injected conditions: agent process death at controlled points, lost acknowledgement, user edits between planning and execution, tool failure, revoked permission, unavailable capability, resource pressure and delayed outcomes. Inject only into the research app/test environment; do not disrupt personal apps. Use real lifecycle tests alongside synthetic replay, and identify which results come from each.

Compare on identical model/data/runtime/quantization:

- Fresh planning at every restart.
- Persistent journal with fixed verification and recovery rules.
- Plan/state validation adapted from the closest prior work.
- A recovery strategy inspired by AndroidReality, with adaptation boundaries disclosed.
- ADF-EA-style evidence/authorization plus BRACE-style cost control, both individually and combined; distinguish reproduced algorithms from original released implementations.
- Compact structured-context reconstruction without lifecycle-aware selection.
- Proposed cost-aware selection; learned selection only if justified.

Include a deterministic workflow engine on the narrow initial suite. If it solves everything more cheaply, the experiment needs genuinely language-dependent, compositional or changed-state tasks before claiming a need for an SLM.

Report verified completion, duplicate/incorrect effects, ambiguous outcomes, recovery latency p50/p95, model calls/tokens, load/prefill/decode overhead, peak RSS/PSS and sustained thermal behavior. Quantized task accuracy must be measured on the deployed artifact. Repeat runs with fixed seeds/task states and confidence intervals. Split results by perturbation, workflow length and device.

Ablate persistent evidence, state validation, resource inputs, model-residency inputs and adaptive selection. Compare single-device tuning against a held-out device. The M35 plus another lower-end stock Android phone is a minimum useful comparison; three devices would support stronger portability evidence. A claim about every phone is unsupported by any such finite sample—publish a measured support range.

## Decisions before major training

1. Reproduce a small set of failures with fixed weights and the simple baselines. If existing techniques solve them at low cost, narrow or change the question before consuming training time.
2. Trace citations from ADF-EA, BRACE, AndroidReality, PlanFence, MobileExplorer and the context/function-calling papers. Check new releases again immediately before submission. An exact overlap requires a pivot or an explicitly incremental study. The literature scope and reading depth are recorded in [literature-review-log.md](literature-review-log.md).
3. Establish a frozen test contract, leakage controls and failure taxonomy. Keep the product roadmap broader than the claim under test.
4. Proceed with controller/model adaptation only after the pilot reveals a measurable weakness. Suggested pilot gate: material latency/inference savings at comparable verified completion, with no increase in duplicate/incorrect effects on the test suite. Set numeric margins before the final evaluation.

If the controller gives no benefit, an honest empirical study of mobile-local failure/cost boundaries may still be useful, but acceptance depends on study depth and venue. Do not turn a negative result into an unsupported “first” claim.

## Deployment boundary

The user chose stock Android, so respect its app sandbox and public capability contracts. [App sandbox](https://source.android.com/docs/security/app-sandbox). Prefer structured APIs/Intents and app-exposed tools. Android 16+ AppFunctions is relevant new infrastructure, but it is experimental and cross-app callers need permission; full agent integration remains restricted in the current documentation. Do not make baseline portability depend on access we do not possess. [AppFunctions](https://developer.android.com/ai/appfunctions).

Accessibility can support a separately enabled research interaction path, but Google Play currently prohibits general autonomous planning/execution through that API, with a specific exception for qualifying accessibility tools. User consent alone does not establish that exception. A research prototype and a distributable product need separate deployment plans. [Current Play policy](https://support.google.com/googleplay/android-developer/answer/10964491?hl=en).
