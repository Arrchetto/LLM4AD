# LLaMEA Log Directory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store all new third-party LLaMEA experiment output under `LLM4AD/GUI/logs/llamea/`.

**Architecture:** Keep the third-party package unchanged. The LLM4AD LLaMEA adapter will redirect the third-party `ExperimentLogger` directory after initialization, using the GUI profiler path to derive the stable `GUI/logs/llamea` parent. Existing `exp-*` directories are preserved.

**Tech Stack:** Python 3.11, unittest, pathlib/os, third-party `llamea`.

---

### Task 1: Define and test the LLaMEA log destination

**Files:**
- Modify: `tests/test_llamea_adapter.py`
- Modify: `llm4ad/method/llamea/llamea.py`

- [ ] **Step 1: Write the failing test**

Add a test that constructs the adapter with a profiler whose `_log_dir` is
`/repo/GUI/logs/<run-name>`, then verifies the LLaMEA logger directory becomes:

```text
/repo/GUI/logs/llamea/exp-<timestamp>-LLaMEA-<model>-
```

Mock the third-party initializer and logger so the test performs no network calls.

- [ ] **Step 2: Run the focused test**

```bash
MPLCONFIGDIR=/tmp/llm4ad-matplotlib .venv/bin/python -m unittest tests.test_llamea_adapter
```

Expected: the new path assertion fails because the adapter does not yet redirect the logger.

- [ ] **Step 3: Implement minimal path redirection**

Add a helper in `llm4ad/method/llamea/llamea.py` that:

```python
profiler_log_dir = os.path.abspath(profiler._log_dir)
gui_logs_dir = os.path.dirname(profiler_log_dir)
llamea_logs_dir = os.path.join(gui_logs_dir, "llamea")
```

Create `llamea_logs_dir`, move the newly created empty/current experiment directory into it, and update `self.logger.dirname`. Only operate on the exact current experiment directory; do not delete or migrate historical directories.

- [ ] **Step 4: Run focused tests**

```bash
MPLCONFIGDIR=/tmp/llm4ad-matplotlib .venv/bin/python -m unittest tests.test_llamea_adapter tests.test_gui_llamea_compat
```

Expected: all focused tests pass.

### Task 2: Verify compatibility

**Files:**
- Test: `tests/test_llamea_adapter.py`
- Test: `tests/test_llamea_evaluation.py`
- Test: `tests/test_gui_llamea_compat.py`
- Test: `tests/test_orienteering_construct.py`

- [ ] **Step 1: Run all tests**

```bash
MPLCONFIGDIR=/tmp/llm4ad-matplotlib .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Expected: all tests pass.

- [ ] **Step 2: Run syntax and diff checks**

```bash
.venv/bin/python -m py_compile llm4ad/method/llamea/llamea.py tests/test_llamea_adapter.py
git diff --check
```

Expected: both commands exit successfully.

- [ ] **Step 3: Verify directory behavior without API calls**

Create a temporary profiler path in a test and verify:

- the `llamea` parent directory is created;
- the experiment directory is nested under it;
- historical experiment directories are untouched;
- non-LLaMEA methods remain unchanged.
