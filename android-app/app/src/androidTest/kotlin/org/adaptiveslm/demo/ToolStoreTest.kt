package org.adaptiveslm.demo

import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

class ToolStoreTest {
    private fun call(id: String, tool: String, args: JSONObject = JSONObject()) = JSONObject()
        .put("version", 1).put("operationId", id).put("tool", tool).put("arguments", args).toString()
    private fun withStore(block: (ToolStore, String) -> Unit) {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val name = "test-${UUID.randomUUID()}.db"
        val store = ToolStore(context, name)
        try { block(store, name) } finally { store.close(); context.deleteDatabase(name) }
    }
    @Test fun lostAcknowledgementReplaySurvivesReopen() = withStore { store, name ->
        val raw = call("create", "tasks.create", JSONObject().put("title", "Fixture"))
        val first = store.execute(raw)
        assertEquals("ok", first.getString("status"))
        store.close()
        ToolStore(InstrumentationRegistry.getInstrumentation().targetContext, name).use { reopened ->
            val replay = reopened.execute(raw)
            assertTrue(replay.getBoolean("replayed"))
            assertEquals(first.getJSONObject("data").getString("id"), replay.getJSONObject("data").getString("id"))
            assertEquals(1, reopened.execute(call("list", "tasks.list")).getJSONObject("data").getJSONArray("items").length())
            assertEquals("operation_id_conflict", reopened.execute(call("create", "tasks.create", JSONObject().put("title", "Other"))).getString("code"))
        }
    }
    @Test fun staleUpdatesAndMissingEntitiesDoNotMutate() = withStore { store, _ ->
        val note = store.execute(call("new", "notes.create", JSONObject().put("title", "Before").put("body", "B"))).getJSONObject("data")
        val id = note.getString("id")
        val args = JSONObject().put("id", id).put("expectedVersion", 1).put("title", "After").put("body", "C")
        assertEquals(2, store.execute(call("update", "notes.update", args)).getJSONObject("data").getInt("version"))
        assertEquals("stale_version", store.execute(call("stale", "notes.update", args)).getString("code"))
        assertEquals("missing_entity", store.execute(call("missing", "notes.read", JSONObject().put("id", "absent"))).getString("code"))
        assertEquals("After", store.execute(call("read", "notes.read", JSONObject().put("id", id))).getJSONObject("data").getString("title"))
    }
    @Test fun cancellationBeforeDispatchLeavesOperationAvailable() = withStore { store, _ ->
        val raw = call("new", "tasks.create", JSONObject().put("title", "Fixture"))
        assertEquals("cancelled", store.execute(raw, AtomicBoolean(true)).getString("code"))
        assertEquals(0, store.execute(call("empty", "tasks.list")).getJSONObject("data").getJSONArray("items").length())
        assertEquals("ok", store.execute(raw).getString("status"))
    }
    @Test fun taskCompletionAndConversationAreDurable() = withStore { store, name ->
        val task = store.execute(call("new", "tasks.create", JSONObject().put("title", "Fixture"))).getJSONObject("data")
        val completed = store.execute(call("complete", "tasks.complete", JSONObject().put("id", task.getString("id")).put("expectedVersion", 1)))
        assertTrue(completed.getJSONObject("data").getBoolean("completed"))
        store.remember("tool", completed.toString())
        store.close()
        ToolStore(InstrumentationRegistry.getInstrumentation().targetContext, name).use { reopened ->
            assertTrue(reopened.history().contains("completed"))
            assertTrue(reopened.execute(call("complete", "tasks.complete", JSONObject().put("id", task.getString("id")).put("expectedVersion", 1))).getBoolean("replayed"))
        }
    }
    @Test fun journalFailureRollsBackEffectAndAllowsRetry() = withStore { store, _ ->
        // Force the failure between effect creation and acknowledgement persistence.
        store.writableDatabase.execSQL("CREATE TRIGGER fail_journal BEFORE INSERT ON journal BEGIN SELECT RAISE(ABORT, 'injected journal failure'); END")
        val raw = call("retry", "tasks.create", JSONObject().put("title", "Atomic fixture"))
        assertThrows(android.database.sqlite.SQLiteException::class.java) { store.execute(raw) }
        store.writableDatabase.execSQL("DROP TRIGGER fail_journal")
        assertEquals(0, store.execute(call("inspect", "tasks.list")).getJSONObject("data").getJSONArray("items").length())
        assertEquals("ok", store.execute(raw).getString("status"))
        assertEquals(1, store.execute(call("inspect-again", "tasks.list")).getJSONObject("data").getJSONArray("items").length())
    }
}
