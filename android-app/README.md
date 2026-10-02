# Offline Android tool harness

This is the first application-owned Android implementation. It executes synthetic notes, task-list and calculator calls locally. The UI is a **scripted fixture console**; no LLM, model import, native inference, broad phone access or autonomous background service is connected yet. No Android permissions, including `INTERNET`, are declared.

## Build and run

Use JDK 21 (JDK 17 is AGP's documented minimum), the installed Android SDK with platform/build-tools 36, and the checked-in Gradle wrapper:

```sh
cd android-app
./gradlew :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest :app:lintDebug
adb -s DEVICE_SERIAL install -r app/build/outputs/apk/debug/app-debug.apk
adb -s DEVICE_SERIAL shell am start -n org.adaptiveslm.demo/.MainActivity
./gradlew :app:connectedDebugAndroidTest
```

Set `JAVA_HOME` and `ANDROID_HOME`, or create an ignored `local.properties` with `sdk.dir`. First build needs access to the official Google/Maven/Gradle registries. Inference and app-tool execution do not need a network. Minimum declared Android API is 26; actual tested device support must be recorded separately. This build contains no native library and makes no model performance claim.

Pinned bootstrap tooling: AGP **9.3.1**, Gradle **9.7.1**, Kotlin supplied by AGP, JVM target **17**. These are the available host versions, not a claim to use every newest patch. [AGP compatibility](https://developer.android.com/build/releases/agp-9-3-0-release-notes) was checked in the browser on 2026-10-02.

## Contract v1

Every call contains exactly `version`, `operationId`, `tool`, `arguments`. The version is integer `1`; operation IDs contain 1–80 ASCII letters, digits, hyphens or underscores. Unknown tools, unknown fields, missing fields, wrong types and oversized inputs fail before mutation.

```json
{"version":1,"operationId":"demo-note-1","tool":"notes.create","arguments":{"title":"Research fixture","body":"Synthetic offline note"}}
```

| Tool | Exact arguments | Outcome |
| --- | --- | --- |
| `calculator.evaluate` | `expression` | Finite decimal arithmetic result; `+ - * / ( )` only. |
| `notes.create` | `title`, `body` | New note UUID, version 1. |
| `notes.read` | `id` | Current note or `missing_entity`. |
| `notes.search` | `query` | Case-insensitive substring matches; empty query lists notes. |
| `notes.update` | `id`, `expectedVersion`, `title`, `body` | Atomic compare-and-update; increments version. |
| `tasks.create` | `title` | New incomplete task UUID, version 1. |
| `tasks.list` | none | Current task state. |
| `tasks.complete` | `id`, `expectedVersion` | Marks complete and increments version, or rejects stale state. |

`status=ok` returns `data`; `status=error` returns `code` and `message`. Valid dispatched calls also return `operationId` and `replayed`. Schema errors and pre-dispatch cancellation consume no journal entry. Arithmetic/state errors from a valid dispatched call are journaled.

App-owned effects and their canonical request/result journal entry commit in **one SQLite transaction**. Same ID and arguments returns the stored result with `replayed=true`; a changed request with the same ID returns `operation_id_conflict`. Read-call replays intentionally return the original snapshot: use a new operation ID to observe current state. IDs are scoped to this app database. This guarantee does not extend to other apps or external side effects.

Cancellation is checked before dispatch and before the effect transaction. Once the effect starts, it commits with its journal or rolls back on error. A late cancellation does not erase committed work. The single background executor keeps database work off the UI thread; shutdown queues database closure after outstanding work. User execution of the displayed app-only call is the current authority mechanism. This is not a complete agent permission system.

## Evaluation

[tool-catalog-v1.json](app/src/main/assets/tool-catalog-v1.json) defines seven synthetic workflows, their allowed scope, requests, dependent call references and observable oracles. `CatalogTest` executes every workflow against a fresh database. These are executor fixtures, **not** evidence of language understanding or a held-out model benchmark.

Separate instrumentation tests check journal replay after database reopen, operation ID conflict, stale versions, missing entities, cancellation, task completion, durable conversation history and rollback after injected journal-write failure. Reopen is not a process-kill/lifecycle benchmark. JVM tests check the bounded arithmetic parser and strict call validation. Keep actual execution results in `docs/research/`.

## Deliberate bootstrap choices and next step

The roadmap still targets Kotlin/Compose and Room. This first console uses framework widgets and `SQLiteOpenHelper` to make executor transactions testable with the installed dependencies. [Android recommends Room](https://developer.android.com/training/data-storage/sqlite); migrate behind the same contract once persistence behavior is verified. No separate Rust integration is needed on this path.

Next: test the APK on the authorized M35, then implement model import/checksum validation and a single-owner JNI inference worker with model-specific prompt templates, streaming, cancellation and measurable release. Preserve the separation between proposed model calls and verified tool outcomes.
