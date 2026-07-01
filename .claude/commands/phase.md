---
description: Run a phase-gated workflow for a feature, new build, or bug remediation.
allowed-tools: [Read, Edit, Write, Bash, Grep, Glob, Task]
---

# Phase-gated workflow

Goal: $ARGUMENTS

Execute in phases, halting between each for user confirmation.

## Phase 1: Explore
Delegate to the `explore` subagent. Read across the relevant parts of the codebase
and produce a structured summary covering:
- Relevant files and their roles
- Existing patterns to follow
- Likely impact surface
- Open questions

STOP. Wait for user confirmation before proceeding.

## Phase 2: Plan
Delegate to the `plan` subagent. Produce a numbered, file-scoped plan with:
- Files to be modified, in order
- Specific changes per file
- Test commands to run after each step
- Success criteria
- Rollback strategy

If the plan agent produces a flawed plan twice, escalate to `hard-plan`.

STOP. Wait for user confirmation before proceeding.

## Phase 3: Execute
Use the `general-purpose` agent (or `execute-local` for non-sensitive code) to
apply the plan one file at a time. After each file, re-read to confirm the
diff matches intent.

## Phase 4: Verify
Delegate to the `verify` subagent. Run the test commands from the plan, parse
output, and report PASS/FAIL with specific failure citations.

If verify fails, return to Phase 2 with the failure context.

## Phase 5: Summarize
Report: files changed, tests run, success criteria met. Stop.
