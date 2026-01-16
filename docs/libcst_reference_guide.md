# LibCST: Comprehensive Reference Guide

> A complete guide to LibCST for experienced Python developers, covering parsing, transformation, metadata analysis, and production patterns.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation and First Contact](#installation-and-first-contact)
3. [Understanding Node Structure](#understanding-node-structure)
4. [The Whitespace Model](#the-whitespace-model)
5. [Visitors: Read-Only Traversal](#visitors-read-only-traversal)
6. [Transformers: Modifying the Tree](#transformers-modifying-the-tree)
7. [The Metadata System](#the-metadata-system)
8. [Matchers: Declarative Pattern Matching](#matchers-declarative-pattern-matching)
9. [Creating Nodes From Scratch](#creating-nodes-from-scratch)
10. [Chaining Transformers](#chaining-transformers)
11. [Industry Example: Import Sorter](#industry-example-import-sorter)
12. [Building a Custom Formatter: Architecture](#building-a-custom-formatter-architecture)
13. [Key Takeaways](#key-takeaways)

---

## Overview

### What Makes LibCST "Concrete"

When Python's `ast` module parses `x = 1 + 2`, it produces nodes representing the *semantic* structure: an assignment target, a binary operation, two integer constants. What it discards: the spaces around `=`, the spaces around `+`, whether you wrote `1` or `0x1` or `0b1`, any trailing comment.

LibCST parses the same code into a tree where every node carries its **surrounding whitespace and formatting** as explicit fields. The `+` operator node knows it has one space before and one space after. Parentheses are explicit nodes, not implicit grouping. String nodes remember their quote style and prefix (`r`, `f`, `b`).

This means you can parse code, modify one specific thing, and serialize back—getting output identical to the input except for your targeted change.

### When to Use LibCST

| Use Case | LibCST Appropriate? |
|----------|---------------------|
| Formatting / code style enforcement | ✅ Yes |
| Codemods / large-scale refactoring | ✅ Yes |
| Linting with autofix capability | ✅ Yes |
| Simple AST analysis (no modification) | ⚠️ Overkill—use `ast` |
| High-performance scenarios (thousands of files) | ⚠️ Consider Ruff |
| Error-tolerant parsing of broken code | ❌ Use Parso instead |

---

## Installation and First Contact

```bash
pip install libcst
```

The entry point is parsing source code into a tree:

```python
import libcst as cst

source = """
def greet(name: str) -> str:
    # Return a greeting
    return f"Hello, {name}!"
"""

module = cst.parse_module(source)
print(type(module))
# <class 'libcst._nodes.module.Module'>

# Round-trip back to source
print(module.code)
# Outputs the original source exactly, including whitespace and comments
```

The `Module` is your root node. Everything else hangs off it.

---

## Understanding Node Structure

Every piece of syntax becomes a node. Let's inspect a simple expression:

```python
import libcst as cst

expr = cst.parse_expression("1 + 2")
print(expr)
```

Output (formatted for readability):
```
BinaryOperation(
    left=Integer(
        value='1',
        lpar=[],
        rpar=[],
    ),
    operator=Add(
        whitespace_before=SimpleWhitespace(
            value=' ',
        ),
        whitespace_after=SimpleWhitespace(
            value=' ',
        ),
    ),
    right=Integer(
        value='2',
        lpar=[],
        rpar=[],
    ),
    lpar=[],
    rpar=[],
)
```

Key observations:

1. **`Integer` stores `value='1'` as a string**, preserving the original representation. If you wrote `0x1`, it would store `'0x1'`.

2. **The `Add` operator carries whitespace fields**. That single space before and after `+` is explicitly captured.

3. **`lpar` and `rpar` are lists**—they hold parentheses if present. `(1 + 2)` would have entries here.

### Exploring Nodes Interactively

```python
import libcst as cst

code = "x = [1, 2, 3]"
module = cst.parse_module(code)

# The module's body is a sequence of statements
stmt = module.body[0]
print(type(stmt))
# <class 'libcst._nodes.statement.SimpleStatementLine'>

# SimpleStatementLine wraps simple statements (assignments, expressions, imports)
# It has a 'body' containing the actual statements
assign = stmt.body[0]
print(type(assign))
# <class 'libcst._nodes.statement.Assign'>

# The assignment has targets and a value
print(assign.targets[0].target.value)  # 'x'
print(type(assign.value))
# <class 'libcst._nodes.expression.List'>

# Dig into the list
for element in assign.value.elements:
    print(element.value.value)
# '1'
# '2'
# '3'
```

### The Node Hierarchy

```
Module
├── body: Sequence[BaseStatement]
│   ├── SimpleStatementLine (for one-liners: assignments, imports, expressions)
│   │   └── body: Sequence[BaseSmallStatement]
│   │       ├── Assign, AnnAssign, AugAssign
│   │       ├── Import, ImportFrom
│   │       ├── Return, Raise, Assert
│   │       ├── Pass, Break, Continue
│   │       └── Expr (expression as statement)
│   │
│   └── CompoundStatement (for blocks)
│       ├── FunctionDef, ClassDef
│       ├── If, For, While, Try, With
│       └── Match (3.10+)
│
├── header: Sequence[EmptyLine]  (blank lines/comments before code)
├── footer: Sequence[EmptyLine]  (blank lines/comments after code)
└── default_indent: str
```

### Node Immutability

Every LibCST node is an immutable dataclass (frozen). You never mutate a tree in place. Instead, transformations return new nodes, and the library efficiently shares unchanged subtrees.

```python
import libcst as cst

original = cst.Name("foo")
modified = original.with_changes(value="bar")

print(original.value)  # 'foo' — unchanged
print(modified.value)  # 'bar'
```

---

## The Whitespace Model

This is the heart of what makes LibCST "concrete." There are several whitespace types:

### SimpleWhitespace

Horizontal whitespace on a single line—spaces and tabs only:

```python
import libcst as cst

ws = cst.SimpleWhitespace("    ")  # Four spaces
print(repr(ws.value))  # '    '
```

Used between tokens on the same line: around operators, after commas, etc.

### ParenthesizedWhitespace

Whitespace that can span lines, but only inside parentheses/brackets/braces:

```python
import libcst as cst

code = """x = (
    1,
    2,
)"""

module = cst.parse_module(code)
tuple_node = module.body[0].body[0].value

# The opening paren has whitespace after it (the newline + indent)
print(type(tuple_node.lpar[0].whitespace_after))
# <class 'libcst._nodes.whitespace.ParenthesizedWhitespace'>
```

`ParenthesizedWhitespace` contains:
- `first_line`: trailing content on the opening line
- `empty_lines`: any blank lines
- `indent`: whether to use the module's default indent
- `last_line`: the whitespace on the final line before content

### TrailingWhitespace

What comes after a statement on the same line—spaces, a comment, and the newline:

```python
import libcst as cst

code = "x = 1  # important\n"
module = cst.parse_module(code)
stmt = module.body[0]

tw = stmt.trailing_whitespace
print(tw.whitespace.value)  # '  ' (two spaces before comment)
print(tw.comment.value)     # '# important'
print(repr(tw.newline))     # Newline(value=None) — None means default newline
```

### EmptyLine

Represents blank lines and standalone comment lines:

```python
import libcst as cst

code = """
# File header

x = 1
"""

module = cst.parse_module(code)

# Header contains the empty lines and comments before first statement
for line in module.header:
    print(f"EmptyLine: comment={line.comment}, whitespace={repr(line.whitespace.value)}")
```

---

## Visitors: Read-Only Traversal

When you need to analyze code without changing it, subclass `CSTVisitor`:

```python
import libcst as cst


class FunctionCounter(cst.CSTVisitor):
    """Counts functions and tracks nesting depth."""

    def __init__(self):
        self.function_count = 0
        self.max_depth = 0
        self._current_depth = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        """
        Called when entering a FunctionDef node.

        Returns:
            bool: True to visit children, False to skip them.
        """
        self.function_count += 1
        self._current_depth += 1
        self.max_depth = max(self.max_depth, self._current_depth)
        return True  # Continue into nested functions

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Called when exiting a FunctionDef node."""
        self._current_depth -= 1


code = """
def outer():
    def middle():
        def inner():
            pass
        return inner
    return middle

def standalone():
    pass
"""

module = cst.parse_module(code)
counter = FunctionCounter()
module.walk(counter)

print(f"Total functions: {counter.function_count}")  # 4
print(f"Maximum nesting: {counter.max_depth}")       # 3
```

### Collecting Information Pattern

```python
import libcst as cst


class StringCollector(cst.CSTVisitor):
    """Collects all string literals with their quote styles."""

    def __init__(self):
        self.strings: list[dict] = []

    def visit_SimpleString(self, node: cst.SimpleString) -> bool:
        quote_char = node.value[0]
        if node.value.startswith('"""') or node.value.startswith("'''"):
            quote_style = "triple"
            quote_char = node.value[:3]
        else:
            quote_style = "single"

        self.strings.append({
            "value": node.value,
            "quote_char": quote_char,
            "quote_style": quote_style,
            "prefix": node.prefix,  # r, f, b, etc.
        })
        return False  # No children to visit

    def visit_FormattedString(self, node: cst.FormattedString) -> bool:
        # f-strings are FormattedString nodes, not SimpleString
        self.strings.append({
            "value": "<f-string>",
            "quote_char": node.start,
            "quote_style": "formatted",
            "prefix": "f",
        })
        return True  # Visit the expressions inside
```

---

## Transformers: Modifying the Tree

Subclass `CSTTransformer` when you need to change code. The key insight: return a new node from `leave_*` methods.

### Basic Transformation: Normalize String Quotes

```python
import libcst as cst


class SingleQuoteNormalizer(cst.CSTTransformer):
    """Converts all double-quoted strings to single quotes."""

    def leave_SimpleString(
        self,
        original_node: cst.SimpleString,
        updated_node: cst.SimpleString,
    ) -> cst.SimpleString:
        """
        Transform string literals from double to single quotes.

        Args:
            original_node: The node as it was before visiting children.
            updated_node: The node after children have been transformed.

        Returns:
            The transformed node.

        Notes:
            The two-argument pattern (original_node, updated_node) exists because
            children are transformed first. For a leaf node like SimpleString,
            they're identical. For compound nodes, updated_node reflects child
            transformations.
        """
        value = updated_node.value
        prefix = updated_node.prefix

        # Handle triple-quoted strings
        if value.startswith('"""'):
            return updated_node

        # Handle regular double-quoted strings
        if value.startswith('"') and not value.startswith('"""'):
            inner = value[1:-1]

            # Skip if inner content contains single quotes (would need escaping)
            if "'" in inner and '"' not in inner:
                return updated_node

            # Convert escapes
            inner = inner.replace("\\'", "\x00")  # Protect existing escapes
            inner = inner.replace('\\"', '"')     # Unescape double quotes
            inner = inner.replace("'", "\\'")     # Escape single quotes
            inner = inner.replace("\x00", "\\'")  # Restore protected escapes

            new_value = f"'{inner}'"
            return updated_node.with_changes(value=prefix + new_value)

        return updated_node
```

### The `with_changes()` Pattern

```python
import libcst as cst


class SpacingNormalizer(cst.CSTTransformer):
    """Ensures exactly one space around binary operators."""

    def leave_BinaryOperation(
        self,
        original_node: cst.BinaryOperation,
        updated_node: cst.BinaryOperation,
    ) -> cst.BinaryOperation:
        # Create new operator with normalized spacing
        new_operator = updated_node.operator.with_changes(
            whitespace_before=cst.SimpleWhitespace(" "),
            whitespace_after=cst.SimpleWhitespace(" "),
        )
        return updated_node.with_changes(operator=new_operator)


code = "x=1+2*3"
module = cst.parse_module(code)
modified = module.visit(SpacingNormalizer())
print(modified.code)  # "x=1 + 2 * 3"
```

### Removing Nodes

Return `cst.RemovalSentinel.REMOVE` to delete a node:

```python
import libcst as cst


class PassRemover(cst.CSTTransformer):
    """Removes unnecessary pass statements from non-empty blocks."""

    def __init__(self):
        self._in_non_empty_block = False

    def visit_IndentedBlock(self, node: cst.IndentedBlock) -> bool:
        stmt_count = len(node.body)
        self._in_non_empty_block = stmt_count > 1
        return True

    def leave_IndentedBlock(
        self,
        original_node: cst.IndentedBlock,
        updated_node: cst.IndentedBlock,
    ) -> cst.IndentedBlock:
        self._in_non_empty_block = False
        return updated_node

    def leave_Pass(
        self,
        original_node: cst.Pass,
        updated_node: cst.Pass,
    ) -> cst.Pass | cst.RemovalSentinel:
        if self._in_non_empty_block:
            return cst.RemovalSentinel.REMOVE
        return updated_node
```

---

## The Metadata System

Raw tree traversal lacks semantic context. Metadata providers add it.

### Position Information

```python
import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider


class FunctionLocator(cst.CSTVisitor):
    """Finds line numbers of all function definitions."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self):
        self.functions: list[tuple[str, int, int]] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        pos = self.get_metadata(PositionProvider, node)
        self.functions.append((
            node.name.value,
            pos.start.line,
            pos.end.line,
        ))
        return True


code = """
def foo():
    pass

def bar():
    x = 1
    y = 2
    return x + y
"""

module = cst.parse_module(code)
wrapper = MetadataWrapper(module)

locator = FunctionLocator()
wrapper.visit(locator)

for name, start, end in locator.functions:
    print(f"{name}: lines {start}-{end}")
# foo: lines 2-3
# bar: lines 5-9
```

Key points:
1. Declare dependencies via `METADATA_DEPENDENCIES` class attribute
2. Wrap the module with `MetadataWrapper`
3. Use `wrapper.visit()` instead of `module.walk()`
4. Access metadata via `self.get_metadata(Provider, node)`

### Scope Analysis

```python
import libcst as cst
from libcst.metadata import MetadataWrapper, ScopeProvider


class UnusedVariableFinder(cst.CSTVisitor):
    """Finds variables that are assigned but never read."""

    METADATA_DEPENDENCIES = (ScopeProvider,)

    def __init__(self):
        self.unused: list[str] = []

    def visit_Module(self, node: cst.Module) -> bool:
        scope = self.get_metadata(ScopeProvider, node)
        self._check_scope(scope)
        return True

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        scope = self.get_metadata(ScopeProvider, node)
        self._check_scope(scope)
        return True

    def _check_scope(self, scope) -> None:
        for assignment in scope.assignments:
            if isinstance(assignment, cst.metadata.Assignment):
                if len(assignment.references) == 0:
                    self.unused.append(assignment.name)
```

### Parent Node Access

```python
import libcst as cst
from libcst.metadata import MetadataWrapper, ParentNodeProvider


class MethodFinder(cst.CSTVisitor):
    """Distinguishes methods (inside class) from functions (standalone)."""

    METADATA_DEPENDENCIES = (ParentNodeProvider,)

    def __init__(self):
        self.methods: list[str] = []
        self.functions: list[str] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        parent = self.get_metadata(ParentNodeProvider, node)

        while parent is not None:
            if isinstance(parent, cst.ClassDef):
                self.methods.append(node.name.value)
                return True
            parent = self.get_metadata(ParentNodeProvider, parent, None)

        self.functions.append(node.name.value)
        return True
```

### Available Metadata Providers

| Provider | Purpose |
|----------|---------|
| `PositionProvider` | Line/column positions for any node |
| `ScopeProvider` | Scope analysis, variable bindings |
| `QualifiedNameProvider` | Fully qualified names for references |
| `ParentNodeProvider` | Navigate upward in the tree |
| `ExpressionContextProvider` | Whether expression is Load, Store, Del |

---

## Matchers: Declarative Pattern Matching

Instead of writing `isinstance()` chains, describe patterns:

```python
import libcst as cst
import libcst.matchers as m

code = "print('hello')"
expr = cst.parse_expression(code)

# Match a call to 'print'
pattern = m.Call(func=m.Name("print"))

if m.matches(expr, pattern):
    print("Found a print() call!")
```

### Complex Matching

```python
import libcst as cst
import libcst.matchers as m


class LogCallFinder(cst.CSTVisitor):
    """Finds all logger.* calls."""

    def visit_Call(self, node: cst.Call) -> bool:
        pattern = m.Call(
            func=m.Attribute(
                value=m.Name("logger"),
                attr=m.OneOf(
                    m.Name("info"),
                    m.Name("debug"),
                    m.Name("warning"),
                    m.Name("error"),
                ),
            )
        )

        if m.matches(node, pattern):
            method = node.func.attr.value
            print(f"Found logger.{method}() call")

        return True
```

### Matcher Utilities

| Function | Purpose |
|----------|---------|
| `m.matches(node, pattern)` | Check if node matches pattern |
| `m.findall(tree, pattern)` | Find all matching nodes |
| `m.replace(tree, pattern, replacement)` | Replace matching nodes |
| `m.DoNotCare()` | Wildcard—matches anything |
| `m.OneOf(...)` | Match any of the given patterns |
| `m.AllOf(...)` | Match all of the given patterns |
| `m.MatchIfTrue(predicate)` | Match if predicate returns True |

---

## Creating Nodes From Scratch

### Parsing Fragments

```python
import libcst as cst

# Parse a statement
stmt = cst.parse_statement("x = calculate_value()")

# Parse an expression
expr = cst.parse_expression("a + b * c")

# These have default whitespace—suitable for generated code
```

### Manual Construction

For precise control (verbose but explicit):

```python
import libcst as cst

# Build: from typing import Optional, List
import_stmt = cst.SimpleStatementLine(
    body=[
        cst.ImportFrom(
            module=cst.Name("typing"),
            names=[
                cst.ImportAlias(name=cst.Name("Optional")),
                cst.ImportAlias(
                    name=cst.Name("List"),
                    comma=cst.MaybeSentinel.DEFAULT,
                ),
            ],
            whitespace_after_from=cst.SimpleWhitespace(" "),
            whitespace_before_import=cst.SimpleWhitespace(" "),
            whitespace_after_import=cst.SimpleWhitespace(" "),
        )
    ]
)

# Usually easier to parse:
import_stmt = cst.parse_statement("from typing import Optional, List")
```

---

## Chaining Transformers

For a multi-rule formatter, compose transformers:

```python
import libcst as cst


class Pipeline:
    """Chains multiple transformers."""

    def __init__(self, *transformers: cst.CSTTransformer):
        self.transformers = transformers

    def transform(self, code: str) -> str:
        module = cst.parse_module(code)

        for transformer in self.transformers:
            module = module.visit(transformer)

        return module.code


# Usage
pipeline = Pipeline(
    TrailingCommaAdder(),
    RemovePassFromNonEmpty(),
    SingleQuoteNormalizer(),
)

result = pipeline.transform(source_code)
```

---

## Industry Example: Import Sorter

A production-quality codemod similar to `isort`:

```python
"""
Import sorter using LibCST.

Organizes imports into groups:
1. Standard library
2. Third-party packages  
3. Local/relative imports

Within each group, imports are sorted alphabetically.
"""

import sys
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import libcst as cst
from libcst.metadata import MetadataWrapper, QualifiedNameProvider


class ImportType(IntEnum):
    """Classification of import statements."""

    FUTURE = 0       # from __future__ import ...
    STDLIB = 1       # import os, from collections import ...
    THIRD_PARTY = 2  # import requests, from django import ...
    LOCAL = 3        # from . import ..., from mypackage import ...


@dataclass
class ImportItem:
    """Represents a single import for sorting purposes."""

    node: cst.SimpleStatementLine
    module_name: str
    import_type: ImportType
    is_from_import: bool

    @property
    def sort_key(self) -> tuple:
        return (
            self.import_type,
            self.is_from_import,
            self.module_name.lower(),
        )


class ImportClassifier:
    """Determines the type of an import based on module name."""

    STDLIB_MODULES: frozenset[str] = frozenset(sys.stdlib_module_names)

    KNOWN_THIRD_PARTY: frozenset[str] = frozenset({
        "requests", "numpy", "pandas", "django", "flask",
        "pytest", "pydantic", "sqlalchemy", "celery", "redis",
        "boto3", "fastapi", "httpx", "aiohttp", "PIL",
    })

    def classify(self, module_name: str, is_relative: bool) -> ImportType:
        if module_name == "__future__":
            return ImportType.FUTURE

        if is_relative:
            return ImportType.LOCAL

        top_level = module_name.split(".")[0]

        if top_level in self.STDLIB_MODULES:
            return ImportType.STDLIB

        if top_level in self.KNOWN_THIRD_PARTY:
            return ImportType.THIRD_PARTY

        return ImportType.THIRD_PARTY


class ImportSorter(cst.CSTTransformer):
    """
    Sorts and groups imports at the top of a module.

    Usage:
        module = cst.parse_module(code)
        modified = module.visit(ImportSorter())
        print(modified.code)
    """

    def __init__(self):
        self.classifier = ImportClassifier()
        self._import_items: list[ImportItem] = []
        self._first_non_import_idx: int | None = None
        self._header_comments: list[cst.EmptyLine] = []

    def visit_Module(self, node: cst.Module) -> bool:
        self._header_comments = list(node.header)
        return True

    def leave_Module(
        self,
        original_node: cst.Module,
        updated_node: cst.Module,
    ) -> cst.Module:
        if not self._import_items:
            return updated_node

        sorted_imports = sorted(
            self._import_items,
            key=lambda x: x.sort_key,
        )

        new_body: list[cst.BaseStatement] = []

        current_type: ImportType | None = None
        for item in sorted_imports:
            if current_type is not None and item.import_type != current_type:
                new_body.append(self._create_blank_line_statement(new_body[-1]))

            stmt = item.node

            if new_body:
                stmt = self._strip_leading_lines(stmt)

            new_body.append(stmt)
            current_type = item.import_type

        if self._first_non_import_idx is not None:
            remaining = list(updated_node.body[self._first_non_import_idx:])

            if remaining and new_body:
                first_remaining = remaining[0]
                remaining[0] = self._ensure_blank_lines_before(first_remaining, count=2)

            new_body.extend(remaining)

        return updated_node.with_changes(body=new_body)

    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine,
    ) -> cst.SimpleStatementLine:
        if len(updated_node.body) != 1:
            self._mark_non_import()
            return updated_node

        stmt = updated_node.body[0]

        if isinstance(stmt, cst.Import):
            self._collect_import(updated_node, stmt)
            return cst.RemoveFromParent()

        if isinstance(stmt, cst.ImportFrom):
            self._collect_from_import(updated_node, stmt)
            return cst.RemoveFromParent()

        self._mark_non_import()
        return updated_node

    def _mark_non_import(self) -> None:
        if self._first_non_import_idx is None:
            self._first_non_import_idx = len(self._import_items)

    def _collect_import(
        self,
        line: cst.SimpleStatementLine,
        stmt: cst.Import,
    ) -> None:
        names = stmt.names
        if isinstance(names, cst.ImportStar):
            module_name = "*"
        else:
            first_alias = names[0]
            if isinstance(first_alias.name, cst.Attribute):
                module_name = self._attribute_to_string(first_alias.name)
            else:
                module_name = first_alias.name.value

        self._import_items.append(ImportItem(
            node=line,
            module_name=module_name,
            import_type=self.classifier.classify(module_name, is_relative=False),
            is_from_import=False,
        ))

    def _collect_from_import(
        self,
        line: cst.SimpleStatementLine,
        stmt: cst.ImportFrom,
    ) -> None:
        is_relative = len(stmt.relative) > 0

        if stmt.module is None:
            module_name = "." * len(stmt.relative)
        elif isinstance(stmt.module, cst.Attribute):
            module_name = self._attribute_to_string(stmt.module)
        else:
            module_name = stmt.module.value

        self._import_items.append(ImportItem(
            node=line,
            module_name=module_name,
            import_type=self.classifier.classify(module_name, is_relative),
            is_from_import=True,
        ))

    def _attribute_to_string(self, attr: cst.Attribute) -> str:
        parts = []
        current = attr
        while isinstance(current, cst.Attribute):
            parts.append(current.attr.value)
            current = current.value
        if isinstance(current, cst.Name):
            parts.append(current.value)
        return ".".join(reversed(parts))

    def _strip_leading_lines(
        self,
        stmt: cst.SimpleStatementLine,
    ) -> cst.SimpleStatementLine:
        return stmt.with_changes(leading_lines=[])

    def _create_blank_line_statement(
        self,
        after_stmt: cst.BaseStatement,
    ) -> cst.EmptyLine:
        return cst.EmptyLine(whitespace=cst.SimpleWhitespace(""))

    def _ensure_blank_lines_before(
        self,
        stmt: cst.BaseStatement,
        count: int,
    ) -> cst.BaseStatement:
        existing_lines = getattr(stmt, "leading_lines", [])

        blank_lines = [
            cst.EmptyLine(whitespace=cst.SimpleWhitespace(""))
            for _ in range(count)
        ]

        if hasattr(stmt, "with_changes"):
            return stmt.with_changes(leading_lines=blank_lines + list(existing_lines))
        return stmt


def sort_imports(code: str) -> str:
    """
    Sort imports in Python source code.

    Args:
        code: Python source code as a string.

    Returns:
        Code with imports sorted and grouped.
    """
    module = cst.parse_module(code)
    modified = module.visit(ImportSorter())
    return modified.code
```

---

## Building a Custom Formatter: Architecture

### Parsing Strategy Options

| Approach | Pros | Cons |
|----------|------|------|
| **LibCST** | Full fidelity, comments preserved | Pure Python (slower), requires valid syntax |
| **AST + tokenize** | Standard library, fast | Comments lost in AST, must stitch back |
| **Token stream** | Simple model | No structural awareness |
| **Parso** | Error recovery, full fidelity | Lower-level API |

### Architecture Pattern: Visitor Pipeline

```
Parse → Analyze → Transform → Emit

1. Parse: cst.parse_module(source) → Module
2. Analyze: MetadataWrapper for scope/position info
3. Transform: Chain of CSTTransformers
4. Emit: module.code → formatted source
```

### Hard Problems in Formatters

1. **Comment attachment**: Which node "owns" a trailing comment? LibCST handles much of this, but edge cases remain.

2. **Line length and breaking**: Constraint satisfaction problem. Black uses "magic trailing comma" heuristic.

3. **Preserving user intent**: Vertical alignment, intentional grouping. Consider `# fmt: off` escape hatches.

---

## Key Takeaways

| Aspect | Guidance |
|--------|----------|
| **When to use LibCST** | Formatting, codemods, linting with autofix |
| **When to avoid** | High-performance scenarios, error-tolerant parsing needed |
| **Learning investment** | Whitespace model: 2-3 days; Metadata system: 1-2 days |
| **Production patterns** | Chain transformers, use metadata for semantics, prefer matchers |
| **Testing codemods** | Snapshot testing—store input/output pairs |

### Ecosystem Tools Using LibCST

- **Bowler**: Facebook's refactoring tool for large-scale codemods
- **autotyping**: Automatic type annotation addition
- **Various internal tools at Instagram/Meta**

---

## Quick Reference

### Parse and Emit

```python
module = cst.parse_module(source)
modified = module.visit(transformer)
output = modified.code
```

### Visitor (Read-Only)

```python
class MyVisitor(cst.CSTVisitor):
    def visit_FunctionDef(self, node) -> bool:
        # Analyze
        return True  # Visit children
```

### Transformer (Modify)

```python
class MyTransformer(cst.CSTTransformer):
    def leave_Name(self, original, updated) -> cst.Name:
        return updated.with_changes(value="new_name")
```

### Metadata

```python
class MyVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider, ScopeProvider)
    
    def visit_Name(self, node) -> bool:
        pos = self.get_metadata(PositionProvider, node)
        return True

wrapper = MetadataWrapper(module)
wrapper.visit(visitor)
```

### Matchers

```python
import libcst.matchers as m

pattern = m.Call(func=m.Name("print"))
if m.matches(node, pattern):
    # Handle match
```
