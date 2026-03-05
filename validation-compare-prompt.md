# Validation: Cleanup Comparing Agent Prompt

Used as step 4 of the validation procedure -- a subagent compares `test.raw.srt` and `test.srt` to assess cleanup quality.

## Prompt

```
Compare test.raw.srt (raw OCR output) and test.srt (LLM-cleaned output) to evaluate cleanup quality. Read both files fully. This is research-only -- do not modify any files.

Perform these checks and report results in the exact sections below:

## 1. Entry counts
- Raw entries: <count>
- Cleaned entries: <count>
- Reduction: <percentage>

## 2. OCR fixes applied (GOOD)
List up to 10 examples of correctly fixed OCR errors, as a table:
| Raw # | Raw text | Cleaned text | Fix type |
Fix types: wrong-char, missing-char, stray-punctuation, garbled-line, line-break

## 3. Noise correctly removed (GOOD)
List entries that were correctly identified as non-dialogue (UI elements, signs, overlay comments, garbled text) and removed. Note how many total.

## 4. Unmerged near-duplicates (BAD)
Find consecutive entries in the CLEANED file where the text is identical or differs only by 1-2 characters (OCR errors). These should have been deduplicated. List each pair with entry numbers and the differing text.

## 5. Legitimate dialogue lost (BAD)
For each removed raw entry, check whether it contained any legible dialogue that should have been kept. Pay special attention to multi-line entries where one line is noise but the other is real dialogue. List any losses.

## 6. Hallucinated corrections (BAD)
Compare each KEEP entry's cleaned text against its raw text. Flag any case where the cleaned version contains words or characters that are NOT plausible OCR corrections of the raw text -- i.e., the model invented content rather than fixing a misread.

## 7. Timing issues
Check for overlapping timestamps or unnatural gaps (>5s between consecutive entries where raw had continuous coverage).

## 8. Summary scorecard
| Category | Count |
|----------|-------|
| OCR errors fixed | <n> |
| Noise entries removed | <n> |
| Near-duplicates merged | <n> |
| Unmerged near-duplicates remaining | <n> |
| Legitimate lines lost | <n> |
| Hallucinated corrections | <n> |
| Timing issues | <n> |

## 9. Pass/Fail
PASS if: 0 hallucinations, <=2 lost lines, 0 unmerged duplicates.
WARN if: 0 hallucinations, <=4 lost lines, <=3 unmerged duplicates.
FAIL otherwise.
State the verdict and the reasons if WARN or FAIL.
```
