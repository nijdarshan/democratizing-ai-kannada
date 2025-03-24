# Word Sense Disambiguation Analysis - Prior vs Posterior Probabilities

## 1. ಗುಂಡಿ (Prior: 81:19)
| Model | Class | Prior | Posterior (Precision) |
|-------|--------|--------|-------------------|
| kannada-bert | hole/pothole | 0.81 | 0.864 |
|              | button | 0.19 | 0.778 |
| muril-base | hole/pothole | 0.81 | 0.856 |
|            | button | 0.19 | 0.857 |
| xl-lexeme | hole/pothole | 0.81 | 0.969 |
|           | button | 0.19 | 0.531 |

## 2. ಅಡಿ (Prior: 64:36)
| Model | Class | Prior | Posterior (Precision) |
|-------|--------|--------|-------------------|
| kannada-bert | feet | 0.64 | 0.970 |
|              | under | 0.36 | 0.940 |
| muril-base | feet | 0.64 | 0.970 |
|            | under | 0.36 | 0.940 |
| xl-lexeme | feet | 0.64 | 0.931 |
|           | under | 0.36 | 0.873 |

## 3. ಮುತ್ತು (Prior: 72:28)
| Model | Class | Prior | Posterior (Precision) |
|-------|--------|--------|-------------------|
| kannada-bert | kiss | 0.72 | 1.000 |
|              | pearl | 0.28 | 0.449 |
| muril-base | kiss | 0.72 | 0.964 |
|            | pearl | 0.28 | 0.420 |
| xl-lexeme | kiss | 0.72 | 0.655 |
|           | pearl | 0.28 | 0.100 |

## 4. ಮತ (Prior: 96:4)
| Model | Class | Prior | Posterior (Precision) |
|-------|--------|--------|-------------------|
| kannada-bert | vote/opinion | 0.96 | 1.000 |
|              | creed | 0.04 | 0.085 |
| muril-base | vote/opinion | 0.96 | 0.938 |
|            | creed | 0.04 | 0.000 |
| xl-lexeme | vote/opinion | 0.96 | 1.000 |
|           | creed | 0.04 | 0.077 |

## 5. ಮಂಡಿ (Prior: 59:41)
| Model | Class | Prior | Posterior (Precision) |
|-------|--------|--------|-------------------|
| kannada-bert | knee | 0.59 | 1.000 |
|              | market | 0.41 | 0.848 |
| muril-base | knee | 0.59 | 1.000 |
|            | market | 0.41 | 0.848 |
| xl-lexeme | knee | 0.59 | 1.000 |
|           | market | 0.41 | 0.639 |