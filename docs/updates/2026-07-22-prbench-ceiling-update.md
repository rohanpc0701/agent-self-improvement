# PRBench Corp-Finance — Ceiling + Open Questions Resolved

**For:** Kartheek (CTO) · **Date:** 2026-07-22 · Short update to the 2026-07-21 memo.

Three things closed since the first memo. Headline unchanged: teacher-built memory lifts the
student **+5.3** on held-out, no fine-tuning.

## The full ladder (12 held-out × k=3)

| Arm | Score |
|---|--:|
| Student alone (floor) | 71.9 |
| Student + memory | 77.2 (**+5.3**) |
| Fable alone (ceiling) | **90.5** |

- Student→teacher gap = **18.6 pts**. Memory closes **28% of it** with prompt text alone.
- Fable is a real ceiling: beats the student on **11/12** tasks; the student never beats it.
- So memory earns a real but **bounded** slice — ~13 pts of headroom remain that in-context
  lessons don't reach. That headroom is the case for the uplift gate and, if needed, LoRA.

## Resolved open questions

1. **The teacher-answer bug is fixed.** Default token budget 4k→12k; teacher calls now warn
   loudly on empty output. Re-ran clean (0 empty answers).
2. **Which memory design wins — settled.** With the real (non-empty) teacher answer added
   as a three-way contrast, the delta *dropped* to +2.5 (regressions doubled, 2→5). The
   simpler **gaps-only** design (student answer + graded missed-criteria, no teacher gold
   answer) is the keeper. Finding: the teacher doesn't need to *solve* the task, only to
   *see where the student failed* — cheaper and better.
3. **Ceiling measured** (above): 90.5.

## Honest bounds (unchanged)
n=12, not yet significance-tested; memory ungated; the +5.3 vs +2.5 gap is within noise so
"gaps-only is *not worse*" is the defensible claim, not "gaps-only is significantly better."

## Next (priority)
Paired bootstrap for a CI → turn the uplift gate on (kills the regressions, chases the 13-pt
headroom) → compute-matched retry arm → full 28 held-out → LoRA if ICL memory plateaus.

Full detail in `docs/updates/2026-07-21-prbench-corpfin-cto.md` (§4, §4a, §5.5, §8).
