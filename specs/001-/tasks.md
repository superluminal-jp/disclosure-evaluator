# Tasks: 複数文書の一括評価

**Input**: Design documents from `/specs/001-/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)

```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands
   → Integration: DB, middleware, logging
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests?
   → All entities have models?
   → All endpoints implemented?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 3.1: Setup

- [x] T001 Create batch processing directory structure (batch_state/, tests/)
- [x] T002 Add batch processing dependencies to requirements.txt
- [x] T003 [P] Configure batch processing linting rules

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

- [x] T004 [P] Contract test batch API endpoints in tests/contract/test_batch_api.py
- [x] T005 [P] Integration test folder-based batch evaluation in tests/integration/test_batch_folder.py
- [x] T006 [P] Integration test mixed document types in tests/integration/test_batch_mixed_types.py
- [x] T007 [P] Integration test batch error handling in tests/integration/test_batch_errors.py
- [x] T008 [P] Integration test batch progress monitoring in tests/integration/test_batch_progress.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)

- [x] T009 [P] BatchEvaluation model in evaluator.py
- [x] T010 [P] BatchDocument model in evaluator.py
- [x] T011 [P] BatchResult model in evaluator.py
- [x] T012 [P] BatchProgress model in evaluator.py
- [x] T013 [P] BatchConfiguration model in evaluator.py
- [x] T014 [P] Supporting enums (BatchStatus, DocumentStatus, ProcessingPhase) in evaluator.py
- [x] T015 BatchEvaluator orchestrator class in evaluator.py
- [x] T016 Document discovery service in evaluator.py
- [x] T017 Batch state persistence service in evaluator.py
- [x] T018 Parallel document processing service in evaluator.py
- [x] T019 Batch progress tracking service in evaluator.py
- [x] T020 Batch result aggregation service in evaluator.py

## Phase 3.4: CLI Integration

- [x] T021 Add --batch CLI argument parsing
- [x] T022 Add --documents CLI argument for file list
- [x] T023 Add --folder CLI argument for folder discovery
- [x] T024 Add --batch-status CLI command
- [x] T025 Add --batch-results CLI command
- [x] T026 Add --resume-batch CLI command
- [x] T027 Add --retry-documents CLI command
- [x] T028 Integrate batch processing with existing CLI workflow

## Phase 3.5: Error Handling & Resilience

- [x] T029 Individual document error tracking
- [x] T030 Batch-level error aggregation
- [x] T031 Retry mechanism for failed documents
- [x] T032 Batch resumption from state files
- [x] T033 Timeout handling for long-running evaluations
- [x] T034 Memory management for large batches

## Phase 3.6: Output & Reporting

- [x] T035 Batch result JSON output formatting
- [x] T036 Batch result summary output formatting
- [x] T037 Batch result CSV output formatting
- [x] T038 Progress indicator implementation
- [x] T039 Batch statistics calculation
- [x] T040 Individual document result preservation

## Phase 3.7: Polish

- [x] T041 [P] Unit tests for batch models in tests/unit/test_batch_models.py
- [x] T042 [P] Unit tests for document discovery in tests/unit/test_document_discovery.py
- [x] T043 [P] Unit tests for state persistence in tests/unit/test_state_persistence.py
- [x] T044 [P] Unit tests for parallel processing in tests/unit/test_parallel_processing.py
- [x] T045 Performance tests for batch processing
- [x] T046 [P] Update README.md with batch processing documentation
- [x] T047 [P] Update CLI help text with batch commands
- [x] T048 Remove code duplication
- [x] T049 Run quickstart.md validation

## Dependencies

- Tests (T004-T008) before implementation (T009-T020)
- T009-T014 (models) before T015-T020 (services)
- T015-T020 (services) before T021-T028 (CLI integration)
- T021-T028 (CLI) before T029-T040 (error handling & output)
- T029-T040 before T041-T049 (polish)

## Parallel Example

```
# Launch T004-T008 together:
Task: "Contract test batch API endpoints in tests/contract/test_batch_api.py"
Task: "Integration test folder-based batch evaluation in tests/integration/test_batch_folder.py"
Task: "Integration test mixed document types in tests/integration/test_batch_mixed_types.py"
Task: "Integration test batch error handling in tests/integration/test_batch_errors.py"
Task: "Integration test batch progress monitoring in tests/integration/test_batch_progress.py"

# Launch T009-T014 together:
Task: "BatchEvaluation model in evaluator.py"
Task: "BatchDocument model in evaluator.py"
Task: "BatchResult model in evaluator.py"
Task: "BatchProgress model in evaluator.py"
Task: "BatchConfiguration model in evaluator.py"
Task: "Supporting enums in evaluator.py"
```

## Notes

- [P] tasks = different files, no dependencies
- Verify tests fail before implementing
- Commit after each task
- Avoid: vague tasks, same file conflicts
- All batch processing extends existing evaluator.py without breaking current functionality

## Task Generation Rules

_Applied during main() execution_

1. **From Contracts**:
   - batch-evaluation-api.yaml → contract test task [P]
   - Each endpoint → implementation task
2. **From Data Model**:
   - BatchEvaluation → model creation task [P]
   - BatchDocument → model creation task [P]
   - BatchResult → model creation task [P]
   - BatchProgress → model creation task [P]
   - Supporting types → model creation tasks [P]
3. **From User Stories**:

   - Folder-based batch evaluation → integration test [P]
   - Mixed document types → integration test [P]
   - Error handling → integration test [P]
   - Progress monitoring → integration test [P]

4. **Ordering**:
   - Setup → Tests → Models → Services → CLI → Error Handling → Output → Polish
   - Dependencies block parallel execution

## Validation Checklist

_GATE: Checked by main() before returning_

- [x] All contracts have corresponding tests
- [x] All entities have model tasks
- [x] All tests come before implementation
- [x] Parallel tasks truly independent
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
