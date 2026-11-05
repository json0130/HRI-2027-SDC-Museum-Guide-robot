# Museum simulator results (200 paired seeds 0..199)

- config hash (sha256 of all effective museum configs): `1929eb86ac450e809b77a50bbd2fc96169f52c32d67dbb239be0fa393a3d01d5`
- git: `e972257cd2969e947e928e2d1447a01bc126da2d` dirty=False
- headline noise=0.1, misread=0.1, turns=24
- Simulated visitors and fake LLM are my assumptions (config/museum_sim.yaml); numbers are per visitor, mean [95% bootstrap CI over seeds].

## Headline: single-visitor scenarios, all six

| metric | fixed | prior-only | profile |
|---|---|---|---|
| wrong_openings | 0.00 [0.00, 0.00] | 0.25 [0.23, 0.27] | 0.25 [0.23, 0.27] |
| right_openings | 0.00 [0.00, 0.00] | 0.75 [0.73, 0.76] | 0.75 [0.73, 0.76] |
| adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 3.24 [3.20, 3.28] |
| right_adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 3.17 [3.12, 3.21] |
| wrong_adapt_opposed | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.07 [0.05, 0.08] |
| wrong_adapt_neutral | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| reversals | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.37 [0.33, 0.40] |
| questions | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 3.24 [3.20, 3.28] |
| offers_declined | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.07 [0.06, 0.09] |
| suspended_writes | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| turns_to_confident | n/a | n/a | 4.78 [4.69, 4.88] |
| confident_dims_frac | n/a | n/a | 0.99 [0.99, 1.00] |
| confident_but_wrong | n/a | n/a | 0.08 [0.06, 0.09] |
| acc_correct | n/a | n/a | 0.98 [0.98, 0.99] |
| acc_wrong | n/a | n/a | 0.01 [0.00, 0.01] |
| acc_no_call | n/a | n/a | 0.01 [0.01, 0.02] |
| neutral_false_pos | n/a | n/a | 0.00 [0.00, 0.00] |
| mae_eff_value | n/a | n/a | 0.07 [0.07, 0.08] |
| like_precision | n/a | n/a | 0.63 [0.62, 0.65] |
| like_recall | n/a | n/a | 0.97 [0.96, 0.98] |
| dislike_precision | n/a | n/a | 0.97 [0.94, 0.99] |
| dislike_recall | n/a | n/a | 0.99 [0.98, 1.00] |

## Scenario: child_correct

| metric | fixed | prior-only | profile |
|---|---|---|---|
| wrong_openings | 0.00 [0.00, 0.00] | 0.11 [0.07, 0.16] | 0.11 [0.07, 0.16] |
| right_openings | 0.00 [0.00, 0.00] | 0.89 [0.84, 0.93] | 0.89 [0.84, 0.93] |
| adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 4.24 [4.14, 4.34] |
| right_adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 4.16 [4.04, 4.26] |
| wrong_adapt_opposed | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.08 [0.04, 0.12] |
| wrong_adapt_neutral | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| reversals | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.42 [0.33, 0.51] |
| questions | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 2.94 [2.84, 3.04] |
| offers_declined | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.08 [0.04, 0.12] |
| suspended_writes | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| turns_to_confident | n/a | n/a | 5.47 [5.29, 5.66] |
| confident_dims_frac | n/a | n/a | 1.00 [0.99, 1.00] |
| confident_but_wrong | n/a | n/a | 0.08 [0.06, 0.10] |
| acc_correct | n/a | n/a | 0.99 [0.98, 1.00] |
| acc_wrong | n/a | n/a | 0.00 [0.00, 0.00] |
| acc_no_call | n/a | n/a | 0.01 [0.00, 0.02] |
| neutral_false_pos | n/a | n/a | 0.00 [0.00, 0.00] |
| mae_eff_value | n/a | n/a | 0.10 [0.09, 0.12] |
| like_precision | n/a | n/a | 0.64 [0.60, 0.68] |
| like_recall | n/a | n/a | 0.97 [0.95, 0.99] |
| dislike_precision | n/a | n/a | 0.00 [0.00, 0.00] |
| dislike_recall | n/a | n/a | n/a |

## Scenario: adult_correct

| metric | fixed | prior-only | profile |
|---|---|---|---|
| wrong_openings | 0.00 [0.00, 0.00] | 0.12 [0.08, 0.17] | 0.12 [0.08, 0.17] |
| right_openings | 0.00 [0.00, 0.00] | 0.88 [0.82, 0.92] | 0.88 [0.82, 0.92] |
| adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 2.25 [2.18, 2.32] |
| right_adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 2.23 [2.15, 2.29] |
| wrong_adapt_opposed | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.03 [0.01, 0.05] |
| wrong_adapt_neutral | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| reversals | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.26 [0.18, 0.34] |
| questions | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 3.06 [2.94, 3.19] |
| offers_declined | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.04 [0.01, 0.07] |
| suspended_writes | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| turns_to_confident | n/a | n/a | 4.42 [4.22, 4.65] |
| confident_dims_frac | n/a | n/a | 1.00 [1.00, 1.00] |
| confident_but_wrong | n/a | n/a | 0.07 [0.04, 0.11] |
| acc_correct | n/a | n/a | 1.00 [1.00, 1.00] |
| acc_wrong | n/a | n/a | 0.00 [0.00, 0.00] |
| acc_no_call | n/a | n/a | 0.00 [0.00, 0.00] |
| neutral_false_pos | n/a | n/a | 0.00 [0.00, 0.00] |
| mae_eff_value | n/a | n/a | 0.02 [0.02, 0.03] |
| like_precision | n/a | n/a | 0.64 [0.60, 0.67] |
| like_recall | n/a | n/a | 0.98 [0.96, 1.00] |
| dislike_precision | n/a | n/a | 0.00 [0.00, 0.00] |
| dislike_recall | n/a | n/a | n/a |

## Scenario: adult_misread_child

| metric | fixed | prior-only | profile |
|---|---|---|---|
| wrong_openings | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| right_openings | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 2.27 [2.20, 2.35] |
| right_adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 2.25 [2.18, 2.32] |
| wrong_adapt_opposed | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.03 [0.01, 0.04] |
| wrong_adapt_neutral | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| reversals | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.28 [0.20, 0.35] |
| questions | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 3.04 [2.90, 3.16] |
| offers_declined | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.03 [0.01, 0.05] |
| suspended_writes | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| turns_to_confident | n/a | n/a | 4.55 [4.32, 4.80] |
| confident_dims_frac | n/a | n/a | 1.00 [1.00, 1.00] |
| confident_but_wrong | n/a | n/a | 0.04 [0.02, 0.07] |
| acc_correct | n/a | n/a | 1.00 [1.00, 1.00] |
| acc_wrong | n/a | n/a | 0.00 [0.00, 0.00] |
| acc_no_call | n/a | n/a | 0.00 [0.00, 0.00] |
| neutral_false_pos | n/a | n/a | 0.00 [0.00, 0.00] |
| mae_eff_value | n/a | n/a | 0.02 [0.02, 0.03] |
| like_precision | n/a | n/a | 0.66 [0.62, 0.70] |
| like_recall | n/a | n/a | 0.99 [0.98, 1.00] |
| dislike_precision | n/a | n/a | n/a |
| dislike_recall | n/a | n/a | n/a |

## Scenario: quiet

| metric | fixed | prior-only | profile |
|---|---|---|---|
| wrong_openings | 0.00 [0.00, 0.00] | 0.11 [0.07, 0.15] | 0.11 [0.07, 0.15] |
| right_openings | 0.00 [0.00, 0.00] | 0.89 [0.84, 0.93] | 0.89 [0.84, 0.93] |
| adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 4.00 [3.87, 4.13] |
| right_adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 3.85 [3.70, 3.99] |
| wrong_adapt_opposed | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.15 [0.10, 0.21] |
| wrong_adapt_neutral | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| reversals | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.50 [0.41, 0.59] |
| questions | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 3.68 [3.61, 3.75] |
| offers_declined | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.10 [0.07, 0.15] |
| suspended_writes | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| turns_to_confident | n/a | n/a | 4.98 [4.72, 5.27] |
| confident_dims_frac | n/a | n/a | 0.97 [0.96, 0.99] |
| confident_but_wrong | n/a | n/a | 0.10 [0.07, 0.12] |
| acc_correct | n/a | n/a | 0.94 [0.91, 0.96] |
| acc_wrong | n/a | n/a | 0.02 [0.01, 0.04] |
| acc_no_call | n/a | n/a | 0.04 [0.02, 0.06] |
| neutral_false_pos | n/a | n/a | 0.00 [0.00, 0.00] |
| mae_eff_value | n/a | n/a | 0.16 [0.14, 0.18] |
| like_precision | n/a | n/a | 0.79 [0.74, 0.84] |
| like_recall | n/a | n/a | 0.88 [0.83, 0.92] |
| dislike_precision | n/a | n/a | 0.00 [0.00, 0.00] |
| dislike_recall | n/a | n/a | n/a |

## Scenario: chatty

| metric | fixed | prior-only | profile |
|---|---|---|---|
| wrong_openings | 0.00 [0.00, 0.00] | 0.07 [0.04, 0.12] | 0.07 [0.04, 0.12] |
| right_openings | 0.00 [0.00, 0.00] | 0.93 [0.89, 0.96] | 0.93 [0.89, 0.96] |
| adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 6.66 [6.54, 6.77] |
| right_adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 6.54 [6.43, 6.64] |
| wrong_adapt_opposed | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.12 [0.07, 0.17] |
| wrong_adapt_neutral | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| reversals | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.76 [0.64, 0.87] |
| questions | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 2.73 [2.62, 2.85] |
| offers_declined | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.18 [0.13, 0.25] |
| suspended_writes | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| turns_to_confident | n/a | n/a | 4.38 [4.23, 4.57] |
| confident_dims_frac | n/a | n/a | 1.00 [1.00, 1.00] |
| confident_but_wrong | n/a | n/a | 0.07 [0.05, 0.10] |
| acc_correct | n/a | n/a | 1.00 [0.99, 1.00] |
| acc_wrong | n/a | n/a | 0.00 [0.00, 0.01] |
| acc_no_call | n/a | n/a | 0.00 [0.00, 0.00] |
| neutral_false_pos | n/a | n/a | n/a |
| mae_eff_value | n/a | n/a | 0.13 [0.11, 0.14] |
| like_precision | n/a | n/a | 0.50 [0.47, 0.54] |
| like_recall | n/a | n/a | 0.98 [0.96, 1.00] |
| dislike_precision | n/a | n/a | 0.00 [0.00, 0.00] |
| dislike_recall | n/a | n/a | n/a |

## Scenario: topic_dislike

| metric | fixed | prior-only | profile |
|---|---|---|---|
| wrong_openings | 0.00 [0.00, 0.00] | 0.09 [0.04, 0.12] | 0.09 [0.04, 0.12] |
| right_openings | 0.00 [0.00, 0.00] | 0.92 [0.88, 0.95] | 0.92 [0.88, 0.95] |
| adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| right_adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| wrong_adapt_opposed | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| wrong_adapt_neutral | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| reversals | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| questions | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 4.00 [4.00, 4.00] |
| offers_declined | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| suspended_writes | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| turns_to_confident | n/a | n/a | n/a |
| confident_dims_frac | n/a | n/a | n/a |
| confident_but_wrong | n/a | n/a | n/a |
| acc_correct | n/a | n/a | n/a |
| acc_wrong | n/a | n/a | n/a |
| acc_no_call | n/a | n/a | n/a |
| neutral_false_pos | n/a | n/a | 0.00 [0.00, 0.00] |
| mae_eff_value | n/a | n/a | 0.00 [0.00, 0.00] |
| like_precision | n/a | n/a | 0.65 [0.61, 0.69] |
| like_recall | n/a | n/a | 1.00 [1.00, 1.00] |
| dislike_precision | n/a | n/a | 1.00 [1.00, 1.00] |
| dislike_recall | n/a | n/a | 0.99 [0.98, 1.00] |

## Swap scenario: swap_detected

| metric | fixed | prior-only | profile |
|---|---|---|---|
| wrong_openings | 0.00 [0.00, 0.00] | 0.09 [0.06, 0.12] | 0.09 [0.06, 0.12] |
| right_openings | 0.00 [0.00, 0.00] | 0.91 [0.88, 0.94] | 0.91 [0.88, 0.94] |
| adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 5.23 [5.13, 5.33] |
| right_adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 5.12 [5.01, 5.22] |
| wrong_adapt_opposed | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.12 [0.09, 0.15] |
| wrong_adapt_neutral | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| reversals | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.59 [0.52, 0.67] |
| questions | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 3.19 [3.13, 3.26] |
| offers_declined | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.11 [0.08, 0.14] |
| suspended_writes | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 2.80 [2.68, 2.92] |
| turns_to_confident | n/a | n/a | 4.86 [4.67, 5.05] |
| confident_dims_frac | n/a | n/a | 0.98 [0.97, 0.99] |
| confident_but_wrong | n/a | n/a | 0.08 [0.06, 0.09] |
| acc_correct | n/a | n/a | 0.97 [0.96, 0.98] |
| acc_wrong | n/a | n/a | 0.01 [0.00, 0.01] |
| acc_no_call | n/a | n/a | 0.02 [0.01, 0.03] |
| neutral_false_pos | n/a | n/a | 0.00 [0.00, 0.00] |
| mae_eff_value | n/a | n/a | 0.15 [0.14, 0.16] |
| like_precision | n/a | n/a | 0.65 [0.62, 0.68] |
| like_recall | n/a | n/a | 0.93 [0.91, 0.95] |
| dislike_precision | n/a | n/a | 0.00 [0.00, 0.00] |
| dislike_recall | n/a | n/a | n/a |

## Swap scenario: swap_silent

| metric | fixed | prior-only | profile |
|---|---|---|---|
| wrong_openings | 0.00 [0.00, 0.00] | 0.12 [0.10, 0.16] | 0.12 [0.10, 0.16] |
| right_openings | 0.00 [0.00, 0.00] | 0.88 [0.84, 0.91] | 0.88 [0.84, 0.91] |
| adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 5.47 [5.34, 5.60] |
| right_adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 4.11 [4.00, 4.22] |
| wrong_adapt_opposed | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.86 [0.80, 0.92] |
| wrong_adapt_neutral | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.51 [0.43, 0.58] |
| reversals | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.37 [1.29, 1.45] |
| questions | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 3.52 [3.46, 3.58] |
| offers_declined | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.53 [0.47, 0.58] |
| suspended_writes | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| turns_to_confident | n/a | n/a | 4.50 [4.38, 4.64] |
| confident_dims_frac | n/a | n/a | 0.99 [0.98, 0.99] |
| confident_but_wrong | n/a | n/a | 0.14 [0.12, 0.16] |
| acc_correct | n/a | n/a | 0.88 [0.86, 0.90] |
| acc_wrong | n/a | n/a | 0.07 [0.06, 0.09] |
| acc_no_call | n/a | n/a | 0.05 [0.04, 0.07] |
| neutral_false_pos | n/a | n/a | 0.88 [0.83, 0.92] |
| mae_eff_value | n/a | n/a | 0.47 [0.45, 0.48] |
| like_precision | n/a | n/a | 0.46 [0.44, 0.48] |
| like_recall | n/a | n/a | 0.88 [0.85, 0.91] |
| dislike_precision | n/a | n/a | 0.00 [0.00, 0.00] |
| dislike_recall | n/a | n/a | n/a |

## Swap scenario: swap_silent_persistent

| metric | fixed | prior-only | profile |
|---|---|---|---|
| wrong_openings | 0.00 [0.00, 0.00] | 0.10 [0.07, 0.14] | 0.10 [0.07, 0.14] |
| right_openings | 0.00 [0.00, 0.00] | 0.90 [0.86, 0.93] | 0.90 [0.86, 0.93] |
| adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 6.21 [6.08, 6.35] |
| right_adaptations | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 5.03 [4.90, 5.16] |
| wrong_adapt_opposed | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.82 [0.76, 0.89] |
| wrong_adapt_neutral | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.36 [0.29, 0.42] |
| reversals | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.47 [1.37, 1.56] |
| questions | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 3.56 [3.51, 3.62] |
| offers_declined | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.59 [0.53, 0.65] |
| suspended_writes | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| turns_to_confident | n/a | n/a | 4.78 [4.61, 4.93] |
| confident_dims_frac | n/a | n/a | 0.99 [0.98, 1.00] |
| confident_but_wrong | n/a | n/a | 0.18 [0.16, 0.21] |
| acc_correct | n/a | n/a | 0.47 [0.45, 0.50] |
| acc_wrong | n/a | n/a | 0.44 [0.42, 0.46] |
| acc_no_call | n/a | n/a | 0.09 [0.07, 0.10] |
| neutral_false_pos | n/a | n/a | 1.00 [1.00, 1.00] |
| mae_eff_value | n/a | n/a | 0.89 [0.87, 0.91] |
| like_precision | n/a | n/a | 0.31 [0.29, 0.33] |
| like_recall | n/a | n/a | 0.64 [0.59, 0.68] |
| dislike_precision | n/a | n/a | 0.00 [0.00, 0.00] |
| dislike_recall | n/a | n/a | n/a |

## Break-even (SENSITIVITY ANALYSIS, not a measured cost)

W/R at which the mode's net benefit equals `fixed` (0). `right` = right adaptation steps + right non-neutral openings, `wrong` = opposed + neutral-truth steps + wrong openings; q = cost per question in units of one right adaptation. Above this ratio `fixed` is better.

| q | prior-only | profile | profile incl. swap scenarios |
|---|---|---|---|
| 0.0 | 2.99 [2.74, 3.26] | 12.31 [11.44, 13.29] | 7.29 [6.98, 7.61] |
| 0.05 | 2.99 [2.74, 3.26] | 11.80 [10.96, 12.74] | 7.04 [6.73, 7.35] |
| 0.1 | 2.99 [2.74, 3.26] | 11.29 [10.48, 12.20] | 6.78 [6.49, 7.09] |

Net benefit per visitor vs fixed (paired seeds), at wrong/right cost ratio W/R, q=0:

| W/R | prior-only | profile |
|---|---|---|
| 1 | 0.50 [0.47, 0.53] | 3.60 [3.55, 3.66] |
| 2 | 0.25 [0.20, 0.29] | 3.28 [3.21, 3.35] |
| 5 | -0.51 [-0.60, -0.41] | 2.33 [2.19, 2.46] |

## Extraction-noise sweep (profile mode, misread at headline)

| noise | wrong_adapt_opposed | wrong_adapt_neutral | reversals | turns_to_confident | acc_correct | acc_wrong | confident_but_wrong |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.06 [0.05, 0.08] | 0.00 [0.00, 0.00] | 0.36 [0.33, 0.40] | 4.68 [4.58, 4.78] | 0.99 [0.98, 0.99] | 0.00 [0.00, 0.01] | 0.07 [0.05, 0.08] |
| 0.1 | 0.07 [0.05, 0.08] | 0.00 [0.00, 0.00] | 0.37 [0.33, 0.40] | 4.78 [4.69, 4.88] | 0.98 [0.98, 0.99] | 0.01 [0.00, 0.01] | 0.08 [0.06, 0.09] |
| 0.2 | 0.07 [0.06, 0.09] | 0.00 [0.00, 0.00] | 0.39 [0.35, 0.42] | 4.84 [4.75, 4.94] | 0.98 [0.97, 0.99] | 0.01 [0.00, 0.01] | 0.09 [0.08, 0.11] |
| 0.3 | 0.09 [0.07, 0.10] | 0.00 [0.00, 0.00] | 0.38 [0.34, 0.42] | 4.91 [4.81, 5.01] | 0.97 [0.96, 0.98] | 0.01 [0.01, 0.02] | 0.11 [0.09, 0.12] |

## Age-misread sweep (noise at headline)

| misread | prior-only wrong_openings | profile wrong_openings | profile wrong_adapt_opposed | break-even W/R prior-only | break-even W/R profile |
|---|---|---|---|---|---|
| 0.0 | 0.17 [0.17, 0.17] | 0.17 [0.17, 0.17] | 0.07 [0.05, 0.08] | 5.00 [5.00, 5.00] | 17.09 [15.93, 18.26] |
| 0.1 | 0.25 [0.23, 0.27] | 0.25 [0.23, 0.27] | 0.07 [0.05, 0.08] | 2.99 [2.74, 3.26] | 12.31 [11.44, 13.29] |
| 0.2 | 0.33 [0.31, 0.35] | 0.33 [0.31, 0.35] | 0.07 [0.05, 0.08] | 2.02 [1.83, 2.23] | 9.63 [8.94, 10.35] |
| 0.3 | 0.42 [0.40, 0.45] | 0.42 [0.40, 0.45] | 0.07 [0.05, 0.08] | 1.37 [1.23, 1.53] | 7.66 [7.17, 8.22] |
