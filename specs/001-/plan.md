# Implementation Plan: 複数文書の一括評価

**Branch**: `001-` | **Date**: 2025-01-04 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-/spec.md`

## Execution Flow (/plan command scope)

```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from file system structure or context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code, or `AGENTS.md` for all other agents).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:

- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary

Batch document evaluation feature that enables simultaneous processing of multiple documents for disclosure compliance evaluation. The system will extend the existing single-document evaluator with parallel processing capabilities, folder-based document discovery, and consolidated result reporting while maintaining individual document traceability and error handling.

## Technical Context

**Language/Version**: Python 3.12 (existing project)  
**Primary Dependencies**: Existing dependencies (OpenAI, Anthropic, boto3, Pydantic) + new: concurrent.futures (built-in), pathlib (built-in), asyncio (built-in)  
**Storage**: File system (existing logs/, outputs/ directories) + new: batch state persistence  
**Testing**: pytest (existing) + new: pytest-asyncio for async testing  
**Target Platform**: Linux/macOS server (existing CLI tool)  
**Project Type**: Single project (CLI tool extension)  
**Performance Goals**: Process 10-50 documents in parallel, <30s total processing time for typical batch  
**Constraints**: Memory efficient processing, LLM API rate limits, existing architecture compatibility  
**Scale/Scope**: 10-100 documents per batch, maintain existing single-document functionality

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

### Clean Architecture Compliance

- ✅ **Dependency Rule**: Batch processing extends existing layers without breaking dependencies
- ✅ **Provider Pattern**: Reuses existing LLM provider abstractions
- ✅ **Modular Monolith**: Batch functionality as new module within existing structure

### SDD/TDD Compliance

- ✅ **Specification First**: Feature spec complete with 15 functional requirements
- ✅ **Test-Driven**: Will generate contract tests and integration tests before implementation
- ✅ **Type Safety**: Pydantic models for batch entities, type hints ≥95%

### Quality Standards

- ✅ **Function Length**: Batch processing functions ≤20 lines
- ✅ **Error Handling**: Result/Either pattern for batch failures
- ✅ **Observability**: Correlation IDs for batch and individual document tracking
- ✅ **Security**: No PII in batch logs, input validation

### Architecture Principles

- ✅ **Single Responsibility**: BatchEvaluator, BatchDocument, BatchResult as separate concerns
- ✅ **Open/Closed**: Extends existing evaluator without modification
- ✅ **Interface Segregation**: Clean batch processing interfaces
- ✅ **Dependency Inversion**: Depends on abstractions, not concrete implementations

## Project Structure

### Documentation (this feature)

```
specs/001-/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)

```
# Single project (DEFAULT) - Extending existing CLI tool
evaluator.py                      # Main evaluator (existing)
├── BatchEvaluator               # New: Batch processing orchestrator
├── BatchDocument                # New: Individual document wrapper
├── BatchResult                  # New: Consolidated results
└── BatchProgress                # New: Progress tracking

tests/
├── test_batch_evaluation.py     # New: Batch processing tests
├── test_batch_entities.py       # New: Entity model tests
└── test_batch_integration.py    # New: End-to-end batch tests

# New directories for batch processing
batch_state/                     # New: Batch processing state persistence
├── active_batches/             # Active batch tracking
└── completed_batches/          # Completed batch results
```

**Structure Decision**: Single project extension - Adding batch processing capabilities to existing CLI tool without breaking current functionality. New batch modules integrate with existing architecture.

## Phase 0: Outline & Research

1. **Extract unknowns from Technical Context** above:

   - Parallel processing patterns for document evaluation
   - File system operations for batch document discovery
   - State persistence for batch processing resumption
   - Error handling strategies for partial batch failures

2. **Generate and dispatch research agents**:

   ```
   Task: "Research Python concurrent.futures patterns for document processing"
   Task: "Research file system discovery patterns for multiple document types"
   Task: "Research state persistence patterns for resumable batch operations"
   Task: "Research error handling patterns for parallel processing failures"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts

_Prerequisites: research.md complete_

1. **Extract entities from feature spec** → `data-model.md`:

   - BatchEvaluation: Collection metadata, processing status, document references
   - BatchDocument: Individual document metadata, processing status, evaluation results
   - BatchResult: Consolidated results, summary statistics, processing metadata
   - BatchProgress: Current state, progress indicators, error information

2. **Generate API contracts** from functional requirements:

   - Batch evaluation endpoints (start, status, results, resume)
   - Document processing endpoints (add, remove, retry)
   - Progress monitoring endpoints (real-time updates)
   - Output OpenAPI schema to `/contracts/`

3. **Generate contract tests** from contracts:

   - Batch evaluation workflow tests
   - Document processing tests
   - Progress monitoring tests
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:

   - Folder-based batch evaluation scenario
   - Mixed document type processing scenario
   - Error handling and recovery scenario
   - Progress monitoring scenario

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/bash/update-agent-context.sh cursor`
   - Add batch processing context
   - Preserve existing agent instructions
   - Update recent changes (keep last 3)

**Output**: data-model.md, /contracts/\*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach

_This section describes what the /tasks command will do - DO NOT execute during /plan_

**Task Generation Strategy**:

- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each contract → contract test task [P]
- Each entity → model creation task [P]
- Each user story → integration test task
- Implementation tasks to make tests pass

**Ordering Strategy**:

- TDD order: Tests before implementation
- Dependency order: Models before services before CLI integration
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 25-30 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation

_These phases are beyond the scope of the /plan command_

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking

_No violations detected - batch processing extends existing architecture cleanly_

## Progress Tracking

_This checklist is updated during execution flow_

**Phase Status**:

- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [ ] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:

- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented

---

_Based on Constitution v2.1.1 - See `/memory/constitution.md`_
