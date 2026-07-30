# Pre-registration — Q1 SELF-PLAN
# COMMIT THIS FILE BEFORE THE FIRST GENERATION CALL. If it isn't committed first, it isn't preregistered.

date: 2026-07-29
branch: feat/q1-self-plan          # pin; never switch mid-run
budget_cap: $10
platform: student=Prime (single stack; claim scoped to student)
          teacher=OpenRouter (plan generation, offline role)
student: qwen/qwen3.6-27b  (reasoning.enabled=false — verify in code)
teacher: z-ai/glm-5.2      (max_tokens>=12000, hard exit on empty content;
                            OpenRouter — GLM empty-content behavior calibrated
                            there in FINDINGS_FINANCE §E; no new serving path)
judge:   openai/gpt-5.2    (assert judge != teacher; verify TOTAL-fallback present on branch)

## Question
When the teacher plan produces its gain, is the mechanism plan STRUCTURE
(student can generate it itself, free to serve) or teacher KNOWLEDGE
(teacher load-bearing, cost story)?

## Arms (same questions, paired)
A. NOPLAN-MATCHED — student writes a ~250-word neutral RESTATEMENT of the
   question (no plan, no strategy words), then answers. This is the
   compute-matched baseline (RALPH confound control).
B. SELF-PLAN — student writes its own <=250-word plan, then answers.
   Plan prompt gives NO domain content, only: "write a brief plan: what is
   being asked, what determines the answer, what steps you will take, what
   mistakes to avoid."
C. TEACHER-PLAN — GLM 5.2 writes <=250-word plan from the question only
   (no rubric — firewall), injected before the student answers.

## Sample & power
- Pilot: 8 questions x 3 arms, k=3 gen repeats, 3 judge passes averaged.
  Gate: per-arm sigma consistent with historical noise floor; no empty
  outputs; spend <= $2.5. If pilot fails mechanically -> fix, re-run pilot.
- Full: 40 validation questions (category-stratified draw from the frozen
  80, seed 42), same protocol. Paired deltas, analysis/bootstrap.py CI.

## Primary metric
recovery = (SELF-PLAN − NOPLAN-MATCHED) / (TEACHER-PLAN − NOPLAN-MATCHED)
computed on paired per-question means.

## Pre-committed branch rule (decided now, not after the number)
- TEACHER-PLAN − NOPLAN-MATCHED CI includes 0  -> the +6 does not replicate
  under matched conditions on this stack; STOP, report, re-scope. recovery
  is undefined and must not be quoted.
- recovery >= 2/3  -> mechanism is STRUCTURE. Next: Q2 (platform check).
- recovery <= 1/3  -> teacher KNOWLEDGE load-bearing. Next: plan-caching
  probe on the critical path.
- 1/3 < recovery < 2/3 -> hybrid. Next: plan-caching, self-plan kept as a
  fallback layer.

## Author prediction (calibration, recorded before data)
prediction: recovery <= 1/3        # per user's stated expectation
confidence: 70 %                   # author (Rohan), recorded before data
notes: reviewer estimate ~50%

## Declared limitations (written now)
- Single stack (Prime). Any positive is scoped until Q2.
- n=40 resolves ~+/-4-5 points at this protocol; recovery ratio inherits
  wide CI if teacher-plan effect is small.
- Restatement control could itself carry a small effect; accepted, declared.
