package org.adaptiveslm.demo

import android.app.Activity
import android.os.Bundle
import android.graphics.Color
import android.view.View
import android.widget.*
import org.json.JSONObject
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/** Deliberately scripted fixture UI; no trained planner is attached yet. */
class MainActivity : Activity() {
    private val worker = Executors.newSingleThreadExecutor()
    private lateinit var store: ToolStore
    private lateinit var output: TextView
    private lateinit var request: EditText
    private lateinit var run: Button
    private var cancellation: AtomicBoolean? = null
    private var destroyed = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = ToolStore(applicationContext)
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 24, 24, 24)
            setBackgroundColor(Color.rgb(247, 248, 252))
        }
        // Apply system insets for target SDK 36 edge-to-edge behavior.
        layout.setOnApplyWindowInsetsListener { view, insets ->
            if (android.os.Build.VERSION.SDK_INT >= 30) {
                val bars = insets.getInsets(android.view.WindowInsets.Type.systemBars())
                view.setPadding(24 + bars.left, 24 + bars.top, 24 + bars.right, 24 + bars.bottom)
            } else {
                @Suppress("DEPRECATION")
                view.setPadding(24 + insets.systemWindowInsetLeft, 24 + insets.systemWindowInsetTop,
                    24 + insets.systemWindowInsetRight, 24 + insets.systemWindowInsetBottom)
            }
            insets
        }
        layout.addView(TextView(this).apply { text = "AdaptiveSLM Lab"; textSize = 25f; setTextColor(Color.rgb(28, 39, 66)) })
        layout.addView(TextView(this).apply {
            text = "Offline tool harness • scripted fixtures\nLLM inference is not connected in this build."
            textSize = 14f
        })
        val fixtures = Spinner(this)
        val labels = listOf("Calculator", "Create note", "Find notes", "Create task", "List tasks")
        fixtures.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, labels)
        layout.addView(fixtures)
        request = EditText(this).apply {
            hint = "Canonical tool call (JSON)"
            textSize = 13f
            minLines = 5
            maxLines = 7
            setHorizontallyScrolling(false)
            gravity = android.view.Gravity.TOP
            inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_FLAG_MULTI_LINE
            isSaveEnabled = true
        }
        layout.addView(request)
        fixtures.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onNothingSelected(parent: AdapterView<*>?) {}
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val (tool, args) = when (position) {
                    0 -> "calculator.evaluate" to JSONObject().put("expression", "(12 + 8) * 3")
                    1 -> "notes.create" to JSONObject().put("title", "Research fixture").put("body", "Synthetic offline note")
                    2 -> "notes.search" to JSONObject().put("query", "Research")
                    3 -> "tasks.create" to JSONObject().put("title", "Run offline evaluation")
                    else -> "tasks.list" to JSONObject()
                }
                request.setText(JSONObject().put("version", 1).put("operationId", UUID.randomUUID().toString())
                    .put("tool", tool).put("arguments", args).toString(2))
            }
        }
        val actions = LinearLayout(this)
        run = Button(this).apply { text = "Execute / replay"; setOnClickListener { execute() } }
        actions.addView(run, LinearLayout.LayoutParams(0, -2, 1f))
        actions.addView(Button(this).apply {
            text = "Cancel"
            setOnClickListener {
                cancellation?.set(true)
                Toast.makeText(context, "Cancellation requested; committed actions remain recorded", Toast.LENGTH_SHORT).show()
            }
        }, LinearLayout.LayoutParams(0, -2, 1f))
        layout.addView(actions)
        layout.addView(TextView(this).apply { text = "Local history • replay uses the same operation ID"; textSize = 14f })
        output = TextView(this).apply { textSize = 13f; setTextIsSelectable(true) }
        layout.addView(ScrollView(this).apply { addView(output) }, LinearLayout.LayoutParams(-1, 0, 1f))
        setContentView(layout)
        worker.execute { val history = store.history(); runOnUiThread { if (!destroyed) output.text = history } }
    }
    private fun execute() {
        val raw = request.text.toString()
        val token = AtomicBoolean(false)
        cancellation = token
        run.isEnabled = false
        worker.execute {
            val history = try {
                val result = store.execute(raw, token)
                store.remember("request", raw)
                store.remember("tool", result.toString(2))
                store.history()
            } catch (e: Exception) { "Execution failed: ${e.javaClass.simpleName}: ${e.message}" }
            runOnUiThread {
                if (!destroyed) { output.text = history; run.isEnabled = true; cancellation = null }
            }
        }
    }
    override fun onDestroy() {
        destroyed = true
        cancellation?.set(true)
        worker.execute { store.close() }
        worker.shutdown()
        super.onDestroy()
    }
}
