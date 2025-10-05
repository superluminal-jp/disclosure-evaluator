# Implementation Plan: API 提供: 他のシステムからの評価リクエスト受付

**Branch**: `003-api` | **Date**: 2025-10-05 | **Spec**: `/specs/003-api/spec.md`
**Input**: Feature specification from `/specs/003-api/spec.md`

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

Create REST API endpoints for external systems to submit document evaluation requests, using AWS Lambda functions for serverless evaluation processing and FastAPI for API layer, supporting both Lambda direct invocation and FastAPI deployment modes.

## Technical Context

**Language/Version**: Python 3.12+ (existing project compatibility)  
**Primary Dependencies**: FastAPI, AWS Lambda Runtime, Pydantic, existing evaluator.py modules  
**Storage**: AWS S3 for batch state persistence, DynamoDB for request tracking (optional)  
**Testing**: pytest, moto (AWS mocking), httpx (API testing)  
**Target Platform**: AWS Lambda (serverless) + FastAPI (containerized/standalone)  
**Project Type**: API service (backend)  
**Performance Goals**: <5s response time for single documents, <30s for batch processing, 100+ concurrent requests  
**Constraints**: Lambda 15min timeout, 10GB memory limit, existing evaluator.py integration  
**Scale/Scope**: 1000+ external systems, 10k+ documents/day, multi-tenant API access

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

### Clean Architecture Compliance

- ✅ **Dependency Rule**: API layer depends on existing evaluator modules (Domain/Application layers)
- ✅ **Provider Pattern**: Existing LLM providers can be reused, API layer as new Infrastructure
- ✅ **Framework Independence**: FastAPI as presentation layer, business logic in existing modules

### Modular Monolith Pattern

- ✅ **Module Boundaries**: API module as new module alongside existing evaluation modules
- ✅ **Interface Contracts**: API endpoints as clear interfaces to existing evaluation services
- ✅ **Minimal Coupling**: API layer calls existing evaluator services, no direct LLM access

### Development Methodology

- ✅ **SDD Compliance**: This plan follows specification-first approach
- ✅ **TDD Ready**: Contract tests will be generated, implementation follows TDD
- ✅ **Type Safety**: FastAPI + Pydantic models for request/response validation

### Quality Standards

- ✅ **Function Length**: API handlers ≤20 lines, delegate to existing services
- ✅ **SOLID Principles**: Single responsibility for each endpoint, dependency injection
- ✅ **Error Handling**: Result pattern for API responses, structured logging

### Security & Compliance

- ✅ **Secrets Management**: AWS Lambda environment variables for API keys
- ✅ **Input Validation**: Pydantic models for request validation
- ✅ **Audit Trails**: Correlation IDs from existing evaluator system

**Status**: ✅ PASS - No constitutional violations detected

## Project Structure

### Documentation (this feature)

```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)

```
# API Module (new)
api/
├── __init__.py
├── main.py                    # FastAPI application entry point
├── lambda_handler.py          # AWS Lambda entry point
├── models/
│   ├── __init__.py
│   ├── requests.py            # Pydantic request models
│   ├── responses.py           # Pydantic response models
│   └── errors.py              # API error models
├── services/
│   ├── __init__.py
│   ├── evaluation_service.py  # Evaluation orchestration
│   ├── batch_service.py       # Batch processing service
│   └── status_service.py      # Status tracking service
├── endpoints/
│   ├── __init__.py
│   ├── evaluation.py          # Single document evaluation endpoints
│   ├── batch.py               # Batch evaluation endpoints
│   ├── status.py              # Status checking endpoints
│   └── health.py              # Health check endpoints
├── middleware/
│   ├── __init__.py
│   ├── auth.py                # Authentication middleware
│   ├── logging.py             # Request logging middleware
│   └── rate_limit.py          # Rate limiting middleware
└── config/
    ├── __init__.py
    ├── settings.py            # API configuration
    └── aws.py                  # AWS-specific configuration

# Existing evaluator modules (reused)
evaluator.py                   # Main evaluation logic (existing)
# ... other existing modules

# Tests (new)
tests/
├── api/
│   ├── __init__.py
│   ├── test_evaluation.py     # Evaluation endpoint tests
│   ├── test_batch.py          # Batch endpoint tests
│   ├── test_status.py         # Status endpoint tests
│   └── test_health.py         # Health endpoint tests
├── services/
│   ├── __init__.py
│   ├── test_evaluation_service.py
│   ├── test_batch_service.py
│   └── test_status_service.py
├── contract/
│   ├── test_api_contracts.py  # OpenAPI contract tests
│   └── test_lambda_contracts.py
└── integration/
    ├── test_api_integration.py
    └── test_lambda_integration.py

# Deployment artifacts
deployment/
├── lambda/
│   ├── requirements.txt       # Lambda-specific dependencies
│   ├── template.yaml          # SAM template
│   └── deploy.sh              # Lambda deployment script
├── docker/
│   ├── Dockerfile             # FastAPI container
│   └── docker-compose.yml     # Local development
└── terraform/
    ├── main.tf               # Infrastructure as code
    └── variables.tf           # Terraform variables
```

**Structure Decision**: API module as new infrastructure layer that integrates with existing evaluator modules. Supports both Lambda (serverless) and FastAPI (containerized) deployment modes through shared service layer.

## Phase 0: Outline & Research

1. **Extract unknowns from Technical Context** above:

   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:

   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts

_Prerequisites: research.md complete_

1. **Extract entities from feature spec** → `data-model.md`:

   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:

   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:

   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:

   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/bash/update-agent-context.sh cursor`
     **IMPORTANT**: Execute it exactly as specified above. Do not add or remove any arguments.
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

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
- Dependency order: Models before services before UI
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 25-30 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation

_These phases are beyond the scope of the /plan command_

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking

_Fill ONLY if Constitution Check has violations that must be justified_

| Violation                  | Why Needed         | Simpler Alternative Rejected Because |
| -------------------------- | ------------------ | ------------------------------------ |
| [e.g., 4th project]        | [current need]     | [why 3 projects insufficient]        |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient]  |

## Progress Tracking

_This checklist is updated during execution flow_

**Phase Status**:

- [ ] Phase 0: Research complete (/plan command)
- [ ] Phase 1: Design complete (/plan command)
- [ ] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:

- [ ] Initial Constitution Check: PASS
- [ ] Post-Design Constitution Check: PASS
- [ ] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented

---

_Based on Constitution v2.1.1 - See `/memory/constitution.md`_
