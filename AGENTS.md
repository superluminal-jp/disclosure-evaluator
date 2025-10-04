# AI Coding Assistant Development Standards

> **Critical Rule**: Specification BEFORE Code. Test BEFORE Implementation. Quality is NON-NEGOTIABLE.

## Decision Tree: When to Write Code

```
User requests code changes
    │
    ├─→ Is it <10 lines + clear behavior + has tests?
    │   └─→ YES: Proceed with implementation
    │   └─→ NO: Continue ↓
    │
    ├─→ Does specification exist (≥95% complete)?
    │   └─→ YES: Continue ↓
    │   └─→ NO: STOP → Create specification first
    │
    ├─→ Is specification approved by stakeholders?
    │   └─→ YES: Proceed with TDD implementation
    │   └─→ NO: STOP → Request approval first
```

## Core Principles

**Philosophy**: Specification-first, test-driven, domain-centric development with zero tolerance for quality degradation.

**Mandatory Approach**: Every task must follow: Specification → Test → Implementation → Documentation

## MANDATORY WORKFLOW

### Phase 1: Specification (BEFORE Code)

**CRITICAL**: NO CODE until this phase is complete (≥95%)

#### Step 1: Requirements Gathering

**Questions AI Must Ask**:

- Who are the stakeholders? (users, maintainers, approvers)
- What is the business value and ROI?
- What are the success criteria and KPIs?
- What are security/compliance requirements? (GDPR, HIPAA, SOC2)
- What are performance requirements? (latency, throughput)
- What edge cases and error scenarios exist?
- What are the risks and mitigation strategies?

#### Step 2: Create Artifacts

- [ ] **Gherkin Scenarios**: Given-When-Then for all behaviors
- [ ] **Domain Model**: Entities, Value Objects, Aggregates, Services
- [ ] **API Contracts**: OpenAPI 3.0 specification with examples
- [ ] **ADRs**: Architecture Decision Records for significant choices
- [ ] **Traceability Matrix**: Requirements → Tests → Code mapping

**Gherkin Template**:

```gherkin
Feature: [Feature Name]
  As a [stakeholder]
  I want [functionality]
  So that [business value]

  Scenario: [Happy Path]
    Given [preconditions]
    When [action]
    Then [expected outcome]
    And [side effects logged/verified]
```

#### Step 3: Validation Gate

- [ ] Requirements coverage ≥95%
- [ ] No contradictions between specs
- [ ] All edge cases documented
- [ ] Security review complete
- [ ] Performance analysis done
- [ ] Stakeholder approval obtained
- [ ] Traceability matrix complete

**Quality Gate**: NO CODE until ALL checks pass

**Exception**: Changes <10 lines with clear behavior, no architectural impact, existing test coverage

### Phase 2: Implementation (TDD + DDD)

**RED-GREEN-REFACTOR Cycle**:

```
1. RED: Write Failing Test
   ├─ Define expected behavior from specification
   ├─ Write minimal test that fails
   ├─ Run test to verify it actually fails
   └─ Commit: "test: add failing test for [feature]"

2. GREEN: Make Test Pass
   ├─ Write minimal code to pass the test
   ├─ No premature optimization
   ├─ Focus on making it work
   └─ Commit: "feat: implement [feature]"

3. REFACTOR: Improve Quality
   ├─ Remove duplication (DRY principle)
   ├─ Improve names (use ubiquitous language)
   ├─ Apply SOLID principles
   ├─ Keep tests green throughout
   └─ Commit: "refactor: improve [component]"

4. DOCUMENT: Update Artifacts
   ├─ Update API documentation
   ├─ Verify traceability matrix
   ├─ Update ADRs if architecture changed
   └─ Commit: "docs: update [documentation]"

5. REPEAT for next feature
```

## AI ENFORCEMENT

**On Code Request**: Gather requirements first (unless pragmatic exception applies)

**Pragmatic Exception**: Changes <10 lines, clear behavior, no architectural impact, existing test coverage

**Reject Code-First**: Request specification creation for any substantial change

## MANDATORY FRAMEWORKS

### SDD (Specification-Driven Development)

- Specification-first before implementation
- ≥95% requirements coverage with traceability
- Contract-by-design (preconditions, postconditions, invariants)
- Property-based testing (Hypothesis/QuickCheck)

### BDD (Behavior-Driven Development)

- Gherkin syntax (Given-When-Then)
- ≥90% business rules covered by executable scenarios
- Living documentation maintained
- Stakeholder collaboration (Three Amigos)

### DDD (Domain-Driven Design)

- **Tactical**: Entities, Value Objects, Aggregates, Domain Services, Repositories
- **Strategic**: Bounded Contexts, Context Maps, Ubiquitous Language
- **Architecture**: Hexagonal/Clean Architecture, dependency inversion
- Domain layer isolated from technical concerns

**Clean Architecture Layers**:

```
┌─────────────────────────────────────┐
│  Presentation (API/UI)              │ ← Adapters (HTTP, CLI)
├─────────────────────────────────────┤
│  Application (Use Cases)            │ ← Orchestration
├─────────────────────────────────────┤
│  Domain (Business Logic)            │ ← Core (NO tech dependencies)
├─────────────────────────────────────┤
│  Infrastructure (DB/External)       │ ← Adapters (PostgreSQL, AWS)
└─────────────────────────────────────┘
      Dependencies point INWARD
```

### TDD (Test-Driven Development)

- **Cycle**: RED (failing test) → GREEN (minimal impl) → REFACTOR
- **Coverage**: ≥80% line, ≥70% branch
- **Pyramid**: 70% unit, 20% integration, 10% E2E
- **Principles**: F.I.R.S.T (Fast, Independent, Repeatable, Self-Validating, Timely)
- **Pattern**: AAA (Arrange-Act-Assert)

### Logging

- **Format**: Structured JSON only
- **Levels**: FATAL/ERROR/WARN/INFO/DEBUG/TRACE
- **Required**: Correlation IDs for tracing
- **Prohibited**: PII, credentials, secrets

### Error Handling

- **Pattern**: Result<T,E> or Either<L,R> (no null returns, no silent failures)
- **Resilience**: Circuit breaker, retry with backoff, timeout
- **Validation**: Fail-fast with early input validation
- **Logging**: All errors logged with context

**Example (Python)**:

```python
from typing import Union
from dataclasses import dataclass

@dataclass
class Success:
    value: str

@dataclass
class Failure:
    error_code: str
    error_message: str
    retry_allowed: bool

Result = Union[Success, Failure]

def process_payment(amount: int) -> Result:
    """Process payment with explicit error handling."""
    if amount <= 0:
        return Failure(
            error_code="INVALID_AMOUNT",
            error_message="Amount must be positive",
            retry_allowed=False
        )
    # Process logic...
    return Success(value="txn_123")
```

## QUALITY GATES (Mandatory)

### Code Quality

| Metric                     | Standard        | Tool                        |
| -------------------------- | --------------- | --------------------------- |
| **SOLID Principles**       | 100% compliance | Manual review               |
| **Function Length**        | ≤20 lines       | Linter                      |
| **Function Parameters**    | ≤3 parameters   | Linter                      |
| **Cyclomatic Complexity**  | ≤10             | radon (Python), ESLint (TS) |
| **Cognitive Complexity**   | ≤15             | SonarQube                   |
| **Test Coverage (Line)**   | ≥80%            | pytest-cov, Jest            |
| **Test Coverage (Branch)** | ≥70%            | pytest-cov, Jest            |
| **Type Hints**             | ≥95%            | mypy strict, tsc strict     |

**SOLID Principles**:

- **S**ingle Responsibility: One class, one reason to change
- **O**pen-Closed: Open for extension, closed for modification
- **L**iskov Substitution: Subtypes must be substitutable
- **I**nterface Segregation: Many specific > one general
- **D**ependency Inversion: Depend on abstractions

### Python

- **Style**: Black + isort + flake8 (PEP 8)
- **Types**: mypy --strict
- **Security**: bandit + safety

### TypeScript

- **Style**: Prettier + ESLint (Airbnb)
- **Types**: strict mode (noImplicitAny, strictNullChecks)
- **Security**: npm audit + snyk

### Documentation

- **API**: 100% public methods documented (JavaDoc/JSDoc)
- **Features**: ≥90% documented
- **Comments**: Intent only (no code translation, no commented code)
- **Freshness**: <30 days for active features

### Error Handling

- **MANDATORY**: Result/Either pattern, circuit breaker for external calls, input validation
- **PROHIBITED**: Silent failures, generic catch-all, null returns, magic numbers

### Logging

- **Distribution**: ERROR <5%, WARN <15%, INFO <30%
- **Format**: 100% JSON with correlation IDs
- **Security**: Zero PII/secrets

## DOCUMENTATION REQUIREMENTS

### Key Stakeholders

- **Business**: PRD, Gherkin scenarios, KPIs, ROI analysis
- **Developers**: README, code docs (JavaDoc/JSDoc), ADRs, setup guides
- **Architects**: C4 diagrams, ADRs, system design, NFRs
- **QA**: Test strategy, test cases, bug templates, automation coverage
- **Operations**: Runbooks, SLOs, IaC, CI/CD, monitoring, incident response
- **Users**: User manuals, FAQ, release notes, accessibility docs (WCAG 2.1 AA)
- **Legal/Compliance**: Privacy policies, licensing, audit trails, regulatory mapping

### Standards

- **Version Control**: All docs in Git
- **API**: OpenAPI 3.0 + auto-generated docs
- **Architecture**: C4 Model + ADRs (MADR format)
- **Metrics**: ≥90% feature coverage, <30 day freshness, zero doc-code discrepancies

## WORKFLOWS

### New Feature

```
1. REQUIREMENTS GATHERING
   ├─ Identify stakeholders (users, maintainers, approvers)
   ├─ Define business value and success criteria
   ├─ Assess security/compliance (GDPR, HIPAA, SOC2)
   ├─ Document performance requirements
   └─ Identify edge cases and risks

2. CREATE SPECIFICATION
   ├─ Write Gherkin scenarios (all behaviors)
   ├─ Design domain model (entities, value objects)
   ├─ Define API contract (OpenAPI 3.0)
   ├─ Create ADRs (architectural decisions)
   └─ Build traceability matrix

3. VALIDATE SPECIFICATION
   ├─ Check completeness (≥95%)
   ├─ Review for contradictions
   ├─ Security review
   ├─ Performance analysis
   └─ Obtain stakeholder approval

4. IMPLEMENT WITH TDD
   ├─ RED: Write failing tests
   ├─ GREEN: Implement features
   ├─ REFACTOR: Improve quality
   └─ DOCUMENT: Update all docs

5. VERIFY QUALITY GATES
   ├─ Run all tests (≥80% coverage)
   ├─ Check complexity (≤10)
   ├─ Verify docs (100% APIs)
   └─ Security scan (bandit/snyk)
```

### Bug Fix

```
1. REPRODUCE BUG
   ├─ Create failing test that reproduces bug
   ├─ Document expected vs actual behavior
   ├─ Identify root cause
   └─ Assess blast radius

2. SPECIFY FIX
   ├─ Document what needs to change
   ├─ Identify affected components
   ├─ Assess risk of regression
   └─ Plan rollback strategy

3. IMPLEMENT FIX
   ├─ Make test pass with minimal changes
   ├─ Verify no regressions (run all tests)
   ├─ Update docs if behavior changed
   └─ Add regression test

4. VERIFY FIX
   ├─ All tests pass
   ├─ Quality gates pass
   ├─ Docs updated
   └─ Stakeholders notified
```

### Refactoring

```
1. ESTABLISH BASELINE
   ├─ Ensure test coverage ≥80%
   ├─ Document current behavior
   ├─ Identify code smells
   └─ Measure current metrics

2. PLAN REFACTORING
   ├─ Define target architecture
   ├─ Create ADR for significant changes
   ├─ Plan incremental steps
   └─ Identify risk mitigation

3. EXECUTE INCREMENTALLY
   ├─ Make small changes (one smell at a time)
   ├─ Keep tests green at each step
   ├─ Commit frequently
   └─ Verify metrics improve

4. VERIFY IMPROVEMENT
   ├─ Measure complexity reduction
   ├─ Verify performance unchanged/improved
   ├─ Update documentation
   └─ Get code review approval
```

## PRE-COMMIT CHECKLIST

- [ ] Specifications complete (≥95%), stakeholder approved, traceable
- [ ] Test coverage ≥80% line, ≥70% branch
- [ ] SOLID principles enforced, no code smells
- [ ] Gherkin scenarios updated and executable
- [ ] Structured JSON logging with correlation IDs
- [ ] Result/Either error handling, circuit breakers on external calls
- [ ] Documentation ≥90%, API 100%, <30 days fresh
- [ ] Domain language used, bounded contexts clean
- [ ] Type hints ≥95%, all quality tools pass (mypy/ESLint/Black/Prettier)

## REJECTION CRITERIA

**IMMEDIATE REJECTION**:

- Code-first development (spec required ≥95%)
- SOLID violations, God classes, primitive obsession, circular dependencies
- Test-last, missing coverage, no Gherkin
- Silent failures, generic catch-all, null returns, magic numbers
- PII in logs, missing correlation IDs, print()/console.log usage
- Missing type hints, `any` types, `except:` without type
- Outdated docs, missing API docs, commented code

## TOOLCHAIN

### Python

**Format**: Black + isort + autoflake | **Lint**: flake8 + pylint | **Types**: mypy strict + pydantic | **Security**: bandit + safety | **Test**: pytest + pytest-cov + hypothesis

**Setup Commands**:

```bash
# Install
pip install black isort autoflake flake8 mypy bandit safety pytest pytest-cov hypothesis

# Format
black . && isort . && autoflake --remove-all-unused-imports -i -r .

# Check
mypy --strict . && pytest --cov=. --cov-report=term-missing && bandit -r . && safety check
```

### TypeScript

**Format**: Prettier | **Lint**: ESLint (Airbnb) + @typescript-eslint | **Types**: tsc strict | **Security**: npm audit + snyk | **Test**: Jest + @testing-library

**Setup Commands**:

```bash
# Install
npm install -D prettier eslint @typescript-eslint/parser jest snyk

# Format
npx prettier --write .

# Check
npx tsc --noEmit && npx eslint . --ext .ts,.tsx && npx jest --coverage && npm audit && npx snyk test
```

### Observability

**Logging**: Structured JSON + ELK/EFK | **Metrics**: Prometheus + Grafana | **Tracing**: OpenTelemetry | **Errors**: Sentry/Rollbar

## QUICK REFERENCE

### Before Code

- Requirements gathered (≥95%)
- Specs created (Gherkin, domain model, ADRs)
- Stakeholder approval
- Traceability matrix

### While Coding

- TDD (Red→Green→Refactor)
- Domain language
- Result/Either errors
- JSON logging + correlation IDs
- Functions ≤20 lines
- Type hints

### After Code

- Tests pass (≥80% line, ≥70% branch)
- Complexity ≤10
- APIs documented (100%)
- Docs updated
- ADRs for arch decisions

### Common Pitfalls → Solutions

| Problem                 | Fix                                 |
| ----------------------- | ----------------------------------- |
| Code-first              | Start with requirements             |
| `print()`/`console.log` | Structured JSON logging             |
| Silent failures         | Result/Either + context logging     |
| Missing types           | Type hints (mypy strict/tsc strict) |
| Long functions          | Extract, ≤20 lines                  |
| Hard-coded values       | Config/env vars                     |
| Unclear names           | Ubiquitous language                 |
| No tests                | TDD first                           |
| God classes             | Single Responsibility               |
| Primitive obsession     | Value objects                       |
| Circular deps           | Dependency inversion                |
| No correlation IDs      | Add to all requests                 |
| PII in logs             | Sanitize, never log secrets         |
| No circuit breakers     | Protect external calls              |

---

## ADR TEMPLATE (MADR Format)

```markdown
# ADR-XXX: [Short Title]

## Status

[Proposed | Accepted | Rejected | Deprecated | Superseded by ADR-YYY]

## Context

What is the issue we're facing? What constraints exist?

## Decision

What did we decide to do? Be specific.

## Consequences

### Positive

- Benefit 1
- Benefit 2

### Negative

- Trade-off 1
- Trade-off 2

### Alternatives Considered

- Alternative A: Rejected because...
- Alternative B: Rejected because...
```

---

## COMMIT MESSAGE STANDARDS

**Format**: `<type>(<scope>): <subject>`

**Types**:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Build process/tooling
- `perf`: Performance improvements

**Examples**:

```
feat(payment): add Stripe integration with circuit breaker

- Implement Stripe payment gateway
- Add circuit breaker for external API calls
- Include retry logic with exponential backoff
- Update API documentation

Closes #123
```

```
fix(auth): prevent session fixation vulnerability

- Regenerate session ID after login
- Add security headers
- Update security tests

BREAKING CHANGE: Session cookies now have httpOnly and secure flags
```

---

## ZERO TOLERANCE POLICY

**Any deviation from these standards constitutes quality degradation and MUST be corrected.**

**AI Assistant Enforcement**:

1. ALWAYS request specifications before writing code (unless <10 lines exception)
2. ALWAYS follow TDD (Red-Green-Refactor)
3. ALWAYS use Result/Either pattern for errors
4. ALWAYS use structured JSON logging
5. ALWAYS add type hints (≥95% coverage)
6. ALWAYS document public APIs (100%)
7. ALWAYS verify quality gates before considering work complete

**Human Developer Enforcement**:

1. NEVER commit without running tests
2. NEVER commit without running linters
3. NEVER commit with failing quality gates
4. NEVER skip code review
5. NEVER merge without stakeholder approval
6. NEVER deploy without documentation update

---

**Last Updated**: 2025-10-01
