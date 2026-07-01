---
description: Hunt down a bug with appropriate tier escalation.
allowed-tools: [Read, Edit, Bash, Grep, Glob, Task]
---

# Bug hunt

Symptom: $ARGUMENTS

## Step 1: Reproduce and characterize
Delegate to `lookup` to find the relevant code path. Use `Bash` to reproduce
the symptom if possible. Classify the bug:
- **Surface** (typo, off-by-one, null check) → continue with current model
- **Structural** (works in isolation, breaks in context) → use `plan` for fix
- **Subtle** (race condition, memory ordering, distributed failure) → escalate to `hard-plan` or `reason`

## Step 2: Locate
Find the specific code responsible. Read it. Read its callers. Read its tests.

## Step 3: Plan the fix
Per the classification above, route to the appropriate planning agent.

## Step 4: Apply and verify
Execute the fix. Delegate to `verify` to confirm the symptom is resolved and
no regression appears.

## Step 5: Report
- Root cause in one sentence
- Files changed
- Test added or modified to catch this in future
- Whether this bug suggests a class of similar bugs to look for
