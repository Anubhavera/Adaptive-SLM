package org.adaptiveslm.demo

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

/** App-owned effects and their acknowledgement commit in the same SQLite transaction. */
class ToolStore(context: Context, name: String = "tools.db") : SQLiteOpenHelper(context, name, null, 1) {
    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("CREATE TABLE entities(id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL, completed INTEGER NOT NULL, version INTEGER NOT NULL)")
        db.execSQL("CREATE TABLE journal(operation_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, result TEXT NOT NULL)")
        db.execSQL("CREATE TABLE conversation(id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT NOT NULL, content TEXT NOT NULL)")
    }
    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        error("A migration is required; never silently discard experiment state")
    }
    @Synchronized
    fun execute(raw: String, cancelled: AtomicBoolean = AtomicBoolean(false)): JSONObject {
        if (cancelled.get()) return failure("cancelled", "Cancelled before dispatch")
        val call = try { ToolCall.parse(raw) } catch (e: Exception) { return failure("invalid_call", e.message ?: "Invalid call") }
        val db = writableDatabase
        db.beginTransaction()
        try {
            db.rawQuery("SELECT fingerprint,result FROM journal WHERE operation_id=?", arrayOf(call.operationId)).use { cursor ->
                if (cursor.moveToFirst()) {
                    return if (cursor.getString(0) == call.fingerprint()) JSONObject(cursor.getString(1)).put("replayed", true)
                    else failure("operation_id_conflict", "Operation ID already belongs to a different request")
                }
            }
            // Cancellation after this boundary does not undo an effect. Return its committed result.
            if (cancelled.get()) return failure("cancelled", "Cancelled before effect transaction")
            val result = dispatch(db, call).put("operationId", call.operationId).put("replayed", false)
            db.insertOrThrow("journal", null, ContentValues().apply {
                put("operation_id", call.operationId); put("fingerprint", call.fingerprint()); put("result", result.toString())
            })
            db.setTransactionSuccessful()
            return result
        } finally { db.endTransaction() }
    }
    private fun dispatch(db: SQLiteDatabase, call: ToolCall): JSONObject {
        val args = call.arguments
        val kind = if (call.tool.startsWith("notes.")) "note" else "task"
        return when (call.tool) {
            "calculator.evaluate" -> try {
                success(JSONObject().put("value", Calculator.calculate(args.getString("expression"))))
            } catch (e: IllegalArgumentException) { failure("arithmetic_error", e.message ?: "Arithmetic error") }
            "notes.create", "tasks.create" -> {
                val id = UUID.randomUUID().toString()
                db.insertOrThrow("entities", null, ContentValues().apply {
                    put("id", id); put("kind", kind); put("title", args.getString("title"))
                    put("body", args.optString("body", "")); put("completed", 0); put("version", 1)
                })
                success(entity(db, id, kind)!!)
            }
            "notes.read" -> entity(db, args.getString("id"), kind)?.let(::success)
                ?: failure("missing_entity", "No such note")
            "notes.search", "tasks.list" -> {
                val rows = JSONArray()
                db.rawQuery("SELECT id,kind,title,body,completed,version FROM entities WHERE kind=? ORDER BY id", arrayOf(kind)).use { cursor ->
                    while (cursor.moveToNext()) {
                        val row = row(cursor)
                        val query = args.optString("query", "")
                        if (call.tool == "tasks.list" || row.getString("title").contains(query, true) || row.getString("body").contains(query, true)) rows.put(row)
                    }
                }
                success(JSONObject().put("items", rows))
            }
            "notes.update", "tasks.complete" -> {
                val id = args.getString("id")
                val before = entity(db, id, kind) ?: return failure("missing_entity", "No such $kind")
                val version = args.getInt("expectedVersion")
                if (before.getInt("version") != version) return failure("stale_version", "Entity changed; read current state before replanning")
                val values = ContentValues().apply {
                    put("version", version + 1)
                    if (kind == "note") { put("title", args.getString("title")); put("body", args.getString("body")) }
                    else put("completed", 1)
                }
                check(db.update("entities", values, "id=? AND kind=? AND version=?", arrayOf(id, kind, version.toString())) == 1)
                success(entity(db, id, kind)!!)
            }
            else -> failure("unsupported_tool", "Unsupported tool")
        }
    }
    private fun entity(db: SQLiteDatabase, id: String, kind: String): JSONObject? =
        db.rawQuery("SELECT id,kind,title,body,completed,version FROM entities WHERE id=? AND kind=?", arrayOf(id, kind)).use {
            if (it.moveToFirst()) row(it) else null
        }
    private fun row(cursor: android.database.Cursor): JSONObject = JSONObject()
        .put("id", cursor.getString(0)).put("kind", cursor.getString(1)).put("title", cursor.getString(2))
        .put("body", cursor.getString(3)).put("completed", cursor.getInt(4) == 1).put("version", cursor.getInt(5))
    @Synchronized
    fun remember(role: String, content: String) {
        writableDatabase.insertOrThrow("conversation", null, ContentValues().apply { put("role", role); put("content", content) })
    }
    @Synchronized
    fun history(): String = readableDatabase.rawQuery("SELECT role,content FROM conversation ORDER BY id DESC LIMIT 30", null).use {
        val lines = mutableListOf<String>()
        while (it.moveToNext()) lines += "${it.getString(0)}: ${it.getString(1)}"
        lines.asReversed().joinToString("\n\n")
    }
    companion object {
        fun success(data: JSONObject) = JSONObject().put("status", "ok").put("data", data)
        fun failure(code: String, message: String) = JSONObject().put("status", "error").put("code", code).put("message", message)
    }
}
