package org.adaptiveslm.demo

import org.json.JSONObject
import org.json.JSONArray
import org.json.JSONTokener

/** Canonical v1 contract shared by scripted fixtures and future model adapters. */
data class ToolCall(val operationId: String, val tool: String, val arguments: JSONObject) {
    fun fingerprint(): String {
        // Values are scalar and validated before reaching the store.
        val ordered = arguments.keys().asSequence().toList().sorted().joinToString(",") {
            val scalar = JSONArray().put(arguments.get(it)).toString().let { json -> json.substring(1, json.length - 1) }
            JSONObject.quote(it) + ":" + scalar
        }
        return "$tool:{$ordered}"
    }
    companion object {
        val fields = mapOf(
            "calculator.evaluate" to setOf("expression"),
            "notes.create" to setOf("title", "body"),
            "notes.read" to setOf("id"),
            "notes.search" to setOf("query"),
            "notes.update" to setOf("id", "expectedVersion", "title", "body"),
            "tasks.create" to setOf("title"),
            "tasks.list" to emptySet(),
            "tasks.complete" to setOf("id", "expectedVersion")
        )
        fun parse(raw: String): ToolCall {
            require(raw.length <= 16384) { "Request too large" }
            val tokener = JSONTokener(raw)
            val obj = tokener.nextValue() as? JSONObject ?: error("Request must be a JSON object")
            require(tokener.nextClean() == '\u0000') { "Unexpected trailing input" }
            require(obj.keys().asSequence().toSet() == setOf("version", "operationId", "tool", "arguments")) { "Unexpected request fields" }
            require(obj.get("version") == 1) { "Only contract version 1 is supported" }
            val op = obj.get("operationId") as? String ?: error("operationId must be a string")
            require(op.matches(Regex("[A-Za-z0-9_-]{1,80}"))) { "Invalid operation ID" }
            val tool = obj.get("tool") as? String ?: error("tool must be a string")
            val expected = fields[tool] ?: error("Unsupported tool")
            val args = obj.getJSONObject("arguments")
            require(args.keys().asSequence().toSet() == expected) { "Arguments must be exactly $expected" }
            expected.forEach { key ->
                val value = args.get(key)
                if (key == "expectedVersion") {
                    require(value is Int && value in 1 until Int.MAX_VALUE) { "expectedVersion must be a positive bounded integer" }
                } else {
                    require(value is String) { "$key must be a string" }
                    val limit = when (key) { "body" -> 4096; "expression" -> 256; "id" -> 80; else -> 160 }
                    require(value.length <= limit) { "$key exceeds $limit characters" }
                    if (key != "body" && key != "query") require(value.isNotBlank()) { "$key must not be blank" }
                }
            }
            return ToolCall(op, tool, args)
        }
    }
}
