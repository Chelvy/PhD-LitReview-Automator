# Research Proposal: The Integration Paradox

**Title:** The Integration Paradox: Why Reliable AI/ML Components Compose into
Unreliable Systems? A Cross-Domain Empirical and Theoretical Analysis

**Last Updated:** 2026-04-09 (auto-maintained by PhD LitReview Automator)

---

## 1. Introduction and Motivation

Modern AI deployment is fundamentally a systems engineering problem. Yet the dominant
paradigm for assessing AI "trustworthiness" remains component-centric: we evaluate
individual models for accuracy, robustness, calibration, fairness, and interpretability
in isolation. This creates what we term the **Integration Paradox**: components that
pass isolated evaluations can compose into systems that are brittle, unpredictable, and
unsafe in deployment.

This paradox manifests through four mechanisms:
1. **Interface assumption misalignment** — semantic, unit, uncertainty representation,
   and data contract violations between components
2. **Timing and concurrency effects** — latency, race conditions, and ordering
   dependencies visible only end-to-end
3. **Feedback loops** — distribution shift and error reinforcement through recursive
   system operation
4. **Human–AI handoff failures** — coupled failure modes at automation boundaries

The EU AI Act (2024–2026 implementation) has elevated this from an academic concern to
a regulatory obligation: high-risk AI systems must undergo lifecycle, system-level risk
management — not just pre-deployment model tests.

---

## 2. Research Questions

**RQ1 (Compositional Formalism):** How should system-level trust be represented and
composed across the AI SDLC, and under what conditions does error/risk amplify
superlinearly?

**RQ2 (Mechanisms & Measurable Indicators):** Which integration-failure mechanisms
recur across domains and SDLC stages, and what measurable indicators predict them?

**RQ3 (Interventions & Auditable Lifecycle Evidence):** Which end-to-end SDLC practices
measurably reduce integration-paradox risk, and how can evidence be packaged into
credible, update-resilient assurance?

---

## 3. Methodology

*[Auto-strengthened by daily pipeline as related methodological papers are found]*

### 3.1 Systematic Literature Review

### 3.2 Cross-Domain Empirical Analysis

### 3.3 Theoretical Framework Development

### 3.4 Validation

---

## 4. Expected Contributions

1. A unified, engineer-usable theory linking AI architecture + interfaces + operational
   drift to system-level trust
2. A validated taxonomy of integration-failure mechanisms (the "Incident Atlas")
3. Architecture-aware, SDLC-integrated indicators for integration risk prediction
4. A set of evidence-composing SDLC practices aligned with EU AI Act lifecycle requirements

---

## 5. References

*[Auto-maintained — populated as papers are included in corpus]*
