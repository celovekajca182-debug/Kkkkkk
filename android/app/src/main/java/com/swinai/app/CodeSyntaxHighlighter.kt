package com.swinai.app

import android.text.Editable
import android.text.style.ForegroundColorSpan
import java.util.regex.Pattern

/** Lightweight on-device token highlighting; it never rewrites the editor text. */
internal object CodeSyntaxHighlighter {
    fun apply(
        editable: Editable,
        language: String,
        colorFor: (String) -> Int
    ) {
        val text = editable.toString()
        val length = text.length
        if (length == 0) return

        editable.getSpans(0, length, ForegroundColorSpan::class.java).forEach {
            editable.removeSpan(it)
        }
        val occupied = BooleanArray(length)

        fun paint(pattern: String, colorKey: String, flags: Int = 0) {
            if (length > 60000) return
            val matcher = Pattern.compile(pattern, flags).matcher(text)
            val color = colorFor(colorKey)
            while (matcher.find()) {
                val start = matcher.start()
                val end = matcher.end()
                if (start >= end) continue
                var overlaps = false
                for (index in start until end) {
                    if (occupied[index]) {
                        overlaps = true
                        break
                    }
                }
                if (overlaps) continue
                editable.setSpan(
                    ForegroundColorSpan(color),
                    start,
                    end,
                    Editable.SPAN_EXCLUSIVE_EXCLUSIVE
                )
                for (index in start until end) occupied[index] = true
            }
        }

        val lang = language.lowercase()
        val commentPattern = when (lang) {
            "python", "gdscript", "ruby", "bash", "shell", "yaml", "toml", "r", "powershell" ->
                """(?m)#.*$"""
            "sql" -> """(?m)--.*$|/\*[\s\S]*?\*/"""
            "html", "xml" -> """<!--[\s\S]*?-->"""
            "css" -> """/\*[\s\S]*?\*/"""
            else -> """//.*?$|/\*[\s\S]*?\*/"""
        }
        val stringPattern = """"(?:""|\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\x60(?:\\.|[^\x60\\])*\x60"""
        paint(stringPattern, "syntaxString")
        paint(commentPattern, "syntaxComment", Pattern.MULTILINE)

        when (lang) {
            "html", "xml" -> paint("""</?[A-Za-z][^>]*>""", "syntaxTag")
            "json" -> paint(""""(?:\\.|[^"\\])*"(?=\\s*:)""", "syntaxTag")
        }

        val keywords = when (lang) {
            "python" -> "and as assert async await break class continue def del elif else except False finally for from global if import in is lambda None nonlocal not or pass raise return True try while with yield"
            "gdscript" -> "and as assert await break class class_name const continue enum extends false for func if in is match not null or pass preload return self signal static super true var while"
            "kotlin" -> "as break class continue do else false for fun if in interface is null object package return super this throw true try typealias val var when while suspend data enum sealed companion override private public internal protected"
            "javascript", "typescript" -> "async await break case catch class const continue debugger default delete do else export extends false finally for from function if import in instanceof let new null of return super switch this throw true try typeof undefined var void while with yield interface type enum implements public private protected readonly static"
            "java" -> "abstract assert boolean break byte case catch char class const continue default do double else enum extends final finally float for if implements import instanceof int interface long native new package private protected public return short static strictfp super switch synchronized this throw throws transient try void volatile while true false null"
            "go" -> "break case chan const continue default defer else fallthrough false for func go goto if import interface map package range return select struct switch true type var"
            "rust" -> "as async await break const continue crate dyn else enum extern false fn for if impl in let loop match mod move mut pub ref return self Self static struct super trait true type union unsafe use where while"
            "sql" -> "all alter as asc begin between by case cascade column commit create cross database default delete desc distinct drop else end exists false foreign from full group having in index inner insert into is join key left like limit not null offset on or order outer primary references right rollback row select set table then true truncate union unique update using values view when where with"
            "bash", "shell" -> "case do done elif else esac export fi for function if in local readonly return set shift then time trap typeset unset until while"
            "php" -> "abstract array as break callable case catch class clone const continue declare default do else elseif empty enddeclare endfor endforeach endif endswitch endwhile extends final finally for foreach function global if implements include instanceof interface namespace new private protected public return static switch throw trait try use var while"
            "swift" -> "associatedtype break case catch class continue default defer do else enum extension false fallthrough for func guard if import in init inout internal is let nil operator private protocol public repeat return self Self static struct subscript super switch throw throws true try typealias var where while"
            "c", "cpp", "csharp" -> "auto bool break case char class const continue default delete do double else enum extern false float for goto if inline int long namespace new nullptr private protected public return short signed sizeof static struct switch this throw true try typedef typename union unsigned using virtual void volatile while"
            else -> "break case class const continue def do else false for function if import in let new null return this throw true try var void while"
        }
        if (keywords.isNotBlank()) {
            val pattern = "\\b(?:" + keywords.split(" ").joinToString("|") { Pattern.quote(it) } + ")\\b"
            paint(pattern, "syntaxKeyword")
        }
        paint("""\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b""", "syntaxNumber")
        if (lang !in setOf("json", "html", "xml", "css", "yaml")) {
            paint("""\b[A-Za-z_][A-Za-z0-9_]*(?=\s*\()""", "syntaxFunction")
        }
    }
}
