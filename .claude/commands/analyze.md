---
description: Read through a codebase to understand how it runs. Exploratory and educational.
allowed-tools: [Read, Grep, Glob]
---

# Codebase analysis

Focus: $ARGUMENTS

Delegate to the `explore` subagent (or `deep-explore` for codebases over 50 files).

Produce a structured report covering:

## System overview
- Entry points and how they're wired
- Top-level architecture (services, modules, layers)
- Data flow from input to output

## Key patterns
- Conventions in use (naming, error handling, async style)
- Notable abstractions and what they hide
- Anything unusual or non-idiomatic

## Dependencies and integrations
- External services
- Internal coupling points

## Risks and rough edges
- Anything that looks like tech debt
- Anything that looks fragile
- Open TODOs or commented-out code worth flagging

## Recommended next reads
- If the user wants to understand X, start with file Y
- If the user wants to modify Z, the surface area is W

Do not propose changes. This is read-only analysis.

