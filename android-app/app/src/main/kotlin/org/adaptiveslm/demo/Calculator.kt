package org.adaptiveslm.demo

/** Bounded arithmetic grammar. No scripting, reflection, functions or exponentiation. */
object Calculator {
    fun calculate(source: String): Double {
        require(source.length in 1..256) { "Expression must contain 1–256 characters" }
        class Parser {
            var position = 0
            var depth = 0
            fun spaces() { while (position < source.length && source[position].isWhitespace()) position++ }
            fun take(c: Char): Boolean {
                spaces()
                if (position < source.length && source[position] == c) { position++; return true }
                return false
            }
            fun checked(value: Double): Double {
                require(value.isFinite()) { "Non-finite arithmetic result" }
                return value
            }
            fun expression(): Double {
                var value = term()
                while (true) value = when {
                    take('+') -> checked(value + term())
                    take('-') -> checked(value - term())
                    else -> return value
                }
            }
            fun term(): Double {
                var value = factor()
                while (true) value = when {
                    take('*') -> checked(value * factor())
                    take('/') -> {
                        val divisor = factor()
                        require(divisor != 0.0) { "Division by zero" }
                        checked(value / divisor)
                    }
                    else -> return value
                }
            }
            fun factor(): Double {
                require(++depth <= 24) { "Expression nesting exceeds 24" }
                try {
                    if (take('+')) return factor()
                    if (take('-')) return checked(-factor())
                    if (take('(')) {
                        val value = expression()
                        require(take(')')) { "Missing closing parenthesis" }
                        return value
                    }
                    spaces()
                    val start = position
                    while (position < source.length && (source[position] in '0'..'9' || source[position] == '.')) position++
                    return checked(source.substring(start, position).toDoubleOrNull()
                        ?: throw IllegalArgumentException("Expected decimal number at $start"))
                } finally { depth-- }
            }
        }
        val parser = Parser()
        val result = parser.expression()
        parser.spaces()
        require(parser.position == source.length) { "Unexpected input at ${parser.position}" }
        return result
    }
}
