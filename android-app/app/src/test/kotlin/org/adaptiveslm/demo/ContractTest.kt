package org.adaptiveslm.demo

import org.junit.Assert.*
import org.junit.Test

class ContractTest {
    @Test fun arithmeticPrecedenceAndUnary() {
        assertEquals(60.0, Calculator.calculate("(12+8)*3"), 0.0)
        assertEquals(-1.5, Calculator.calculate("-3 / +2"), 0.0)
        assertEquals(7.0, Calculator.calculate("1+2*3"), 0.0)
    }
    @Test fun invalidArithmeticCannotRunCode() {
        listOf("1/0", "NaN", "1e309", "System.exit(0)", "2**3", "1+", "(2", "(".repeat(25)+"1"+")".repeat(25), "1".repeat(257)).forEach {
            assertThrows(IllegalArgumentException::class.java) { Calculator.calculate(it) }
        }
    }
    private fun call(args: String = "{}", tool: String = "tasks.list", version: String = "1") =
        """{"version":$version,"operationId":"test_1","tool":"$tool","arguments":$args}"""
    @Test fun canonicalIdentityIgnoresArgumentOrder() {
        val first = ToolCall.parse(call("""{"title":"A","body":"B"}""", "notes.create"))
        val second = ToolCall.parse(call("""{"body":"B","title":"A"}""", "notes.create"))
        assertEquals(first.fingerprint(), second.fingerprint())
    }
    @Test fun rejectUnsupportedAndUnexpectedArguments() {
        listOf(call("""{"extra":1}"""), call(tool="shell.execute"), call(version="2"), call(version="\"1\""),
            call("""{"id":"x","expectedVersion":1.5}""", "tasks.complete"),
            call("""{"title":null}""", "tasks.create"), call("""{"title":""}""", "tasks.create"),
            call() + " trailing", "[]").forEach {
            assertThrows(Exception::class.java) { ToolCall.parse(it) }
        }
    }
}
