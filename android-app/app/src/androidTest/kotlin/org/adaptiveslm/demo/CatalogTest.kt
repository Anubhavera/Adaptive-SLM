package org.adaptiveslm.demo

import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

class CatalogTest {
    private fun at(root: Any, path: String): Any {
        var value = root
        path.split('.').forEach { part ->
            value = when (val current = value) {
                is JSONObject -> current.get(part)
                is JSONArray -> if (part == "length") current.length() else current.get(part.toInt())
                else -> error("Cannot traverse $part in $path")
            }
        }
        return value
    }
    @Test fun allSyntheticWorkflowsMatchObservableOracles() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val catalog = context.assets.open("tool-catalog-v1.json").bufferedReader().use { JSONObject(it.readText()) }
        val workflows = catalog.getJSONArray("workflows")
        for (index in 0 until workflows.length()) {
            val workflow = workflows.getJSONObject(index)
            val name = "catalog-${UUID.randomUUID()}.db"
            val results = JSONObject()
            val store = ToolStore(context, name)
            try {
                val steps = workflow.getJSONArray("steps")
                for (i in 0 until steps.length()) {
                    val step = steps.getJSONObject(i)
                    val call = step.getJSONObject("call")
                    val args = call.getJSONObject("arguments")
                    args.keys().asSequence().toList().forEach { key ->
                        val reference = args.optJSONObject(key)
                        if (reference != null) args.put(key, at(results, reference.getString("\$ref")))
                    }
                    val result = store.execute(call.toString(), AtomicBoolean(step.optBoolean("cancelled", false)))
                    results.put(step.getString("name"), result)
                    val expected = step.getJSONObject("expect")
                    expected.keys().forEach { path ->
                        val actual = at(result, path)
                        val wanted = expected.get(path)
                        val message = "${workflow.getString("id")}/${step.getString("name")}: $path"
                        if (wanted is Number && actual is Number) assertEquals(message, wanted.toDouble(), actual.toDouble(), 0.0)
                        else assertEquals(message, wanted, actual)
                    }
                }
            } finally { store.close(); context.deleteDatabase(name) }
        }
    }
}
