<!--
Sync Impact Report:
Version change: N/A → 1.0.0
Initial constitution creation for Disclosure Evaluator project
Core principles derived from:
  - AGENTS.md governance rules
  - evaluator.py implementation patterns
  - README.md project documentation
  - docs/AGENTS.md documentation standards
Templates requiring updates:
  ⚠ plan-template.md - PENDING (to be created)
  ⚠ spec-template.md - PENDING (to be created)
  ⚠ tasks-template.md - PENDING (to be created)
Follow-up TODOs:
  - Create .specify/templates/ directory structure
  - Establish baseline testing framework
  - Configure automated quality gates
-->

# Project Constitution: Disclosure Evaluator

**Version**: 1.0.0
**Ratification Date**: 2025-10-04
**Last Amended**: 2025-10-04

---

## Preamble

This constitution establishes the foundational governance principles for the **Disclosure Evaluator** project—an advanced information disclosure law evaluation system for Japan's Freedom of Information Act. These principles are non-negotiable and must be upheld by all contributors, whether human or AI-assisted development agents.

---

## Article I: Project Identity

### Section 1.1: Name and Purpose

**Project Name**: Disclosure Evaluator (情報公開法評価システム)

**Purpose**: Provide a robust, LLM-powered evaluation system that assesses information disclosure possibilities under Japanese Freedom of Information Act (情報公開法) with comprehensive legal analysis and structured outputs.

**Mission**: Empower administrative decision-makers with reliable, traceable, and legally-grounded disclosure evaluations through AI-assisted analysis while maintaining human oversight and accountability.

### Section 1.2: Target Audience

- **Primary**: Japanese administrative organizations and government agencies
- **Secondary**: Legal professionals, researchers, compliance officers
- **Tertiary**: Open-source contributors and AI/ML developers

---

## Article II: Core Architecture Principles

### Section 2.1: Clean Architecture Mandate

**The Dependency Rule**: All dependencies MUST point inward. Outer layers depend on inner layers; inner layers remain framework-agnostic.

**Layer Structure**:

- **Domain Layer**: Pydantic models (EvaluationStep, CriterionEvaluation, DisclosureEvaluationResult)
- **Application Layer**: Business logic (StepEvaluator, CriterionEvaluator, ResultAggregator)
- **Infrastructure Layer**: LLM providers (OpenAIProvider, AnthropicProvider, BedrockProvider)
- **Presentation Layer**: CLI interface and output formatters

**Provider Pattern Enforcement**: All external integrations (LLM APIs, file systems) MUST be abstracted through provider interfaces to ensure testability and flexibility.

**Rationale**: Clean architecture ensures long-term maintainability, enables independent evolution of layers, and prevents vendor lock-in with LLM providers.

### Section 2.2: Modular Monolith Pattern

**Current Architecture**: Modular Monolith with clear module boundaries

- **Evaluation Core**: Step-by-step evaluation logic
- **Provider Module**: Multi-provider LLM abstraction
- **Configuration Module**: Centralized config management
- **Logging Module**: Structured observability

**Module Independence**: Each module MUST maintain clear interfaces and minimal coupling. Cross-module communication MUST occur through well-defined contracts.

**Future Evolution**: Architecture MAY evolve to microservices only when operational complexity justifies distributed systems (team size >50, independent scaling requirements).

**Rationale**: Modular monolith provides deployment simplicity while maintaining architectural discipline for potential future scaling.

---

## Article III: Development Methodology Principles

### Principle 1: Specification-Driven Development (SDD)

**Declaration**: Specification BEFORE Code. Implementation follows documented requirements with ≥95% specification coverage.

**Mandatory Artifacts**:

- Gherkin scenarios (Given-When-Then) for all user-facing behaviors
- API contracts (function signatures, pre/postconditions, invariants)
- Architecture Decision Records (ADRs) for significant technical choices
- Traceability matrix linking requirements → tests → code

**Validation Gate**: No code merges without corresponding specification documentation.

**Rationale**: SDD prevents scope creep, reduces rework, enables parallel development, and creates living documentation. With AI-assisted development, specifications serve as a safety net against AI hallucinations and regressions.

### Principle 2: Test-Driven Development (TDD)

**Declaration**: Tests BEFORE Implementation. All production code MUST be driven by failing tests.

**Coverage Requirements**:

- Line coverage: ≥80%
- Branch coverage: ≥70%
- Test pyramid: 70% unit tests, 20% integration tests, 10% end-to-end tests

**RED-GREEN-REFACTOR Cycle**:

1. **RED**: Write a failing test that defines desired behavior
2. **GREEN**: Implement minimal code to make the test pass
3. **REFACTOR**: Improve code quality while maintaining green tests

**Testing Standards**:

- Tests MUST be F.I.R.S.T (Fast, Independent, Repeatable, Self-Validating, Timely)
- Use pytest with fixtures for dependency injection
- Property-based testing with Hypothesis for comprehensive validation

**Zero Tolerance**: No code commits without corresponding tests. No exceptions.

**Rationale**: TDD ensures correctness, prevents regressions, enables confident refactoring, and serves as executable documentation. Critical for AI-assisted development where generated code quality varies.

### Principle 3: Type Safety and Validation

**Declaration**: Static typing and runtime validation are NON-NEGOTIABLE.

**Type Coverage Requirements**:

- Type hints: ≥95% of all functions, methods, and class attributes
- Strict mode: mypy strict or Pyright enabled
- Pydantic models for all data structures crossing boundaries

**Validation Strategy**:

- Input validation at system boundaries (CLI args, API calls, file I/O)
- Domain model validation through Pydantic with custom validators
- Fail-fast principle: Invalid data rejected immediately with clear error messages

**Enforcement**:

- Pre-commit hooks for type checking
- CI/CD pipeline blocks merges on type errors
- Runtime validation exceptions logged with correlation IDs

**Rationale**: Type safety prevents runtime errors, improves IDE support, serves as living documentation, and catches bugs early. Essential for LLM integration where response formats vary.

### Principle 4: Structured Observability

**Declaration**: All system behavior MUST be observable through structured logging and tracing.

**Logging Standards**:

- Format: Structured JSON only (no plain text logs)
- Required fields: timestamp, level, message, correlation_id, component
- Severity levels: DEBUG (development), INFO (business events), WARN (recoverable), ERROR (failures), CRITICAL (system outages)

**Correlation ID Mandate**:

- Every evaluation request MUST have a unique correlation ID (format: `eval_YYYYMMDD_HHMMSS`)
- Correlation IDs MUST propagate through all components (evaluator → criterion evaluator → step evaluator → LLM provider)
- Log aggregation MUST be correlation-ID-aware for request tracing

**Security and Privacy**:

- ZERO TOLERANCE for PII in logs (personal names, addresses, phone numbers)
- API keys and secrets MUST NEVER appear in logs
- Input text MAY be logged at DEBUG level only, with truncation

**Performance Monitoring**:

- Log execution times for operations >100ms
- Track LLM API latency and token usage
- Monitor parallel evaluation efficiency

**Rationale**: Structured observability enables production debugging, performance optimization, audit compliance, and incident response. Critical for LLM-based systems where failures are often non-deterministic.

### Principle 5: Quality Gates and Code Standards

**Declaration**: All code MUST meet objective quality standards before merge.

**Code Complexity Limits**:

- Function length: ≤20 lines (excluding docstrings/comments)
- Cognitive complexity: ≤15 (enforced by linters)
- Cyclomatic complexity: ≤10
- Class size: ≤300 lines

**SOLID Principles Compliance**: 100% adherence required

- Single Responsibility: Each class/function has one reason to change
- Open/Closed: Open for extension, closed for modification
- Liskov Substitution: Subtypes must be substitutable for base types
- Interface Segregation: No fat interfaces with unused methods
- Dependency Inversion: Depend on abstractions, not concretions

**Tooling and Automation**:

- **Linter**: Ruff (10-100x faster than Black/Flake8)
- **Type Checker**: Pyright or mypy strict mode
- **Test Runner**: pytest with pytest-cov
- **Security Scanner**: bandit for vulnerability detection

**Pre-commit Hooks**: Auto-formatting, linting, type checking, test execution

**CI/CD Quality Gates**:

- All tests passing (no flaky tests tolerated)
- Coverage thresholds met (≥80% line, ≥70% branch)
- No linter errors or warnings
- No type errors
- No high-severity security vulnerabilities

**Rationale**: Automated quality gates prevent technical debt accumulation, reduce code review burden, and maintain consistent codebase health. Essential for multi-contributor and AI-assisted development.

### Principle 6: Error Handling and Resilience

**Declaration**: All error conditions MUST be anticipated, handled gracefully, and logged comprehensively.

**Error Handling Strategy**:

- **Fail-fast validation**: Invalid inputs rejected at system boundaries
- **Result/Either pattern**: Functions return `Result[T, Error]` instead of raising exceptions for expected failures
- **Circuit breakers**: LLM API failures trigger circuit breakers with exponential backoff
- **Fallback mechanisms**: Parallel evaluation degradation (parallel → sequential on thread pool exhaustion)

**Exception Hierarchy**:

- Domain-specific exceptions (e.g., `EvaluationError`, `ProviderError`, `ConfigurationError`)
- Rich error context (correlation IDs, input summaries, recovery suggestions)
- Never swallow exceptions silently (always log with ERROR level)

**Resilience Patterns**:

- Retry with exponential backoff and jitter for transient failures
- Timeout enforcement on all external API calls
- Graceful degradation (e.g., fallback step results on parallel evaluation failure)

**Rationale**: Robust error handling ensures system reliability in production, reduces incident response time, and improves user experience. Critical for LLM integrations where API failures and rate limits are common.

### Principle 7: Security and Compliance

**Declaration**: Security is a first-class concern. All code MUST follow security best practices.

**Secrets Management**:

- API keys MUST be stored in environment variables or secure vaults (never in code/config files)
- `.env` files MUST be git-ignored
- Use python-dotenv for local development
- Production secrets managed via AWS Secrets Manager / Azure Key Vault / HashiCorp Vault

**Input Validation**:

- All user inputs sanitized and validated
- JSON parsing with schema validation (Pydantic models)
- Prevent injection attacks (though LLM prompts are controlled, evaluate for prompt injection risks)

**Dependency Security**:

- Regular dependency audits with `pip-audit` or `safety`
- Pin dependency versions in `requirements.txt`
- Automated vulnerability scanning in CI/CD
- Use Socket.dev for supply chain attack detection

**Compliance Considerations**:

- PII handling: Follow GDPR principles (minimize collection, secure storage, right to erasure)
- Audit trails: Correlation IDs enable request tracing for compliance audits
- Data retention: Define and enforce retention policies for logs and evaluation results

**Rationale**: Security breaches and compliance failures undermine trust and can cause legal/financial consequences. LLM systems handling legal information require extra diligence.

---

## Article IV: Documentation Standards

### Section 4.1: Code Documentation

**Docstring Requirements**:

- All public functions/classes/methods MUST have docstrings
- Format: Google-style or NumPy-style (consistent project-wide)
- Include: purpose, parameters (with types), return values, exceptions raised, examples

**Inline Comments**:

- Explain WHY, not WHAT (code should be self-documenting)
- Document complex algorithms, legal logic, and non-obvious decisions
- Keep comments synchronized with code changes

### Section 4.2: Architecture Documentation

**Required Documentation**:

- README.md: Project overview, installation, usage examples, architecture summary
- AGENTS.md: AI coding assistant governance and best practices
- docs/AGENTS.md: Detailed documentation standards
- ADRs (docs/adr/): Architecture Decision Records for significant choices

**Living Documentation**:

- Update documentation alongside code changes
- Specification documents serve as living requirements
- Test cases serve as executable documentation

**Freshness Requirements**:

- README.md: Update within 30 days of feature changes
- API documentation: Update within 7 days of interface changes
- ADRs: Immutable once ratified (create new ADRs for amendments)

---

## Article V: Governance and Amendment Process

### Section 5.1: Roles and Responsibilities

**Project Maintainers**: Responsible for code review, architecture decisions, and constitution amendments.

**Contributors**: Must follow constitution principles; violations result in PR rejection.

**AI Coding Assistants**: MUST adhere to all constitution principles. Human oversight required for merges.

### Section 5.2: Amendment Procedure

**Proposal**: Any maintainer may propose amendments via pull request.

**Review Period**: Minimum 7 days for community feedback.

**Approval**: Requires consensus among active maintainers (or majority vote if consensus unachievable).

**Versioning**: Constitution follows semantic versioning:

- **MAJOR**: Backward-incompatible governance changes (e.g., removing principles)
- **MINOR**: New principles or sections added
- **PATCH**: Clarifications, typo fixes, non-semantic refinements

**Propagation**: All dependent templates (.specify/templates/) MUST be updated to reflect amendments.

### Section 5.3: Compliance Review

**Frequency**: Quarterly reviews of constitution adherence.

**Metrics**:

- Test coverage percentages
- Type hint coverage
- Linter violations
- Documentation freshness

**Corrective Actions**: Non-compliance triggers technical debt remediation sprints.

### Section 5.4: Conflict Resolution

**Conflicts between Principles**: Specification-first and type safety principles take precedence over velocity.

**Emergency Overrides**: Security vulnerabilities may bypass normal approval processes but MUST be documented retroactively.

---

## Article VI: AI-Assisted Development Guardrails

### Section 6.1: AI Tool Usage

**Approved Uses**:

- Code generation following specifications
- Test case generation
- Documentation writing
- Code review assistance

**Prohibited Uses**:

- Committing AI-generated code without human review
- Bypassing quality gates "because AI generated it"
- Accepting AI suggestions without verification

### Section 6.2: AI-Generated Code Requirements

**All AI-generated code MUST**:

- Follow all constitution principles (no exceptions)
- Include human-written specifications as input context
- Be reviewed by human maintainers before merge
- Pass all automated quality gates
- Include tests (AI-generated tests are acceptable if validated)

### Section 6.3: Specification as AI Safety Net

**Rationale**: Specifications constrain AI behavior and prevent hallucinations.

**Process**:

1. Human writes specification (Gherkin scenarios, contracts, ADRs)
2. AI generates implementation and tests
3. Human reviews against specification
4. Automated gates verify quality standards

---

## Article VII: Adoption and Ratification

This constitution was ratified on **2025-10-04** by the project maintainers.

All existing code is grandfathered under legacy status. Future changes MUST comply with this constitution.

A 6-month transition period is granted for full compliance with Principles 2 (TDD coverage requirements) and 3 (Type hints ≥95%).

---

**Signed**:
Project Maintainers, Disclosure Evaluator Project

**Last Constitutional Review**: 2025-10-04
**Next Scheduled Review**: 2026-01-04
