### Table 1: `true_sense` vs `predicted_sense`

| Model                    | Word   |   Sense |   Prior |   Precision (predicted_sense) |   Recall (predicted_sense) |
|:-------------------------|:-------|--------:|--------:|------------------------------:|---------------------------:|
| l3cube-pune/kannada-bert | ಅಡಿ     |       1 |    0.36 |                          0.97 |                       0.89 |
| l3cube-pune/kannada-bert | ಅಡಿ     |       2 |    0.64 |                          0.94 |                       0.98 |
| l3cube-pune/kannada-bert | ಗುಂಡಿ     |       1 |    0.8  |                          0.85 |                       0.98 |
| l3cube-pune/kannada-bert | ಗುಂಡಿ     |       2 |    0.2  |                          0.75 |                       0.3  |
| l3cube-pune/kannada-bert | ಮಂಡಿ     |       1 |    0.6  |                          1    |                       0.88 |
| l3cube-pune/kannada-bert | ಮಂಡಿ     |       2 |    0.4  |                          0.85 |                       1    |
| l3cube-pune/kannada-bert | ಮತ     |       1 |    0.96 |                          1    |                       0.86 |
| l3cube-pune/kannada-bert | ಮತ     |       2 |    0.04 |                          0.24 |                       1    |
| l3cube-pune/kannada-bert | ಮುತ್ತು    |       2 |    0.28 |                          0.41 |                       1    |
| l3cube-pune/kannada-bert | ಮುತ್ತು    |       1 |    0.72 |                          1    |                       0.43 |
| google/muril-base-cased  | ಅಡಿ     |       1 |    0.36 |                          0.97 |                       0.86 |
| google/muril-base-cased  | ಅಡಿ     |       2 |    0.64 |                          0.93 |                       0.98 |
| google/muril-base-cased  | ಗುಂಡಿ     |       1 |    0.8  |                          0.83 |                       0.99 |
| google/muril-base-cased  | ಗುಂಡಿ     |       2 |    0.2  |                          0.8  |                       0.2  |
| google/muril-base-cased  | ಮಂಡಿ     |       1 |    0.6  |                          1    |                       0.88 |
| google/muril-base-cased  | ಮಂಡಿ     |       2 |    0.4  |                          0.85 |                       1    |
| google/muril-base-cased  | ಮತ     |       1 |    0.96 |                          1    |                       0.85 |
| google/muril-base-cased  | ಮತ     |       2 |    0.04 |                          0.22 |                       1    |
| google/muril-base-cased  | ಮುತ್ತು    |       2 |    0.28 |                          0.41 |                       1    |
| google/muril-base-cased  | ಮುತ್ತು    |       1 |    0.72 |                          1    |                       0.43 |
| pierluigic/xl-lexeme     | ಅಡಿ     |       1 |    0.36 |                          0.93 |                       0.75 |
| pierluigic/xl-lexeme     | ಅಡಿ     |       2 |    0.64 |                          0.87 |                       0.97 |
| pierluigic/xl-lexeme     | ಗುಂಡಿ     |       1 |    0.8  |                          0.79 |                       0.88 |
| pierluigic/xl-lexeme     | ಗುಂಡಿ     |       2 |    0.2  |                          0.09 |                       0.05 |
| pierluigic/xl-lexeme     | ಮಂಡಿ     |       1 |    0.6  |                          0.97 |                       0.57 |
| pierluigic/xl-lexeme     | ಮಂಡಿ     |       2 |    0.4  |                          0.6  |                       0.98 |
| pierluigic/xl-lexeme     | ಮತ     |       1 |    0.96 |                          0.98 |                       0.9  |
| pierluigic/xl-lexeme     | ಮತ     |       2 |    0.04 |                          0.17 |                       0.5  |
| pierluigic/xl-lexeme     | ಮುತ್ತು    |       2 |    0.28 |                          0.09 |                       0.07 |
| pierluigic/xl-lexeme     | ಮುತ್ತು    |       1 |    0.72 |                          0.67 |                       0.72 |


### Table 2: `true_sense` vs `llm_annotation`

| Model                    | Word   |   Sense |   Prior |   Precision (llm_annotation) |   Recall (llm_annotation) |
|:-------------------------|:-------|--------:|--------:|-----------------------------:|--------------------------:|
| l3cube-pune/kannada-bert | ಅಡಿ     |       1 |    0.36 |                         0.97 |                      0.81 |
| l3cube-pune/kannada-bert | ಅಡಿ     |       2 |    0.64 |                         0.9  |                      0.98 |
| l3cube-pune/kannada-bert | ಗುಂಡಿ     |       1 |    0.8  |                         0.97 |                      0.92 |
| l3cube-pune/kannada-bert | ಗುಂಡಿ     |       2 |    0.2  |                         0.75 |                      0.9  |
| l3cube-pune/kannada-bert | ಮಂಡಿ     |       1 |    0.6  |                         1    |                      0.92 |
| l3cube-pune/kannada-bert | ಮಂಡಿ     |       2 |    0.4  |                         0.89 |                      1    |
| l3cube-pune/kannada-bert | ಮತ     |       1 |    0.96 |                         1    |                      0.93 |
| l3cube-pune/kannada-bert | ಮತ     |       2 |    0.04 |                         0.36 |                      1    |
| l3cube-pune/kannada-bert | ಮುತ್ತು    |       2 |    0.28 |                         0.81 |                      0.79 |
| l3cube-pune/kannada-bert | ಮುತ್ತು    |       1 |    0.72 |                         0.92 |                      0.93 |
| google/muril-base-cased  | ಅಡಿ     |       1 |    0.36 |                         0.97 |                      0.81 |
| google/muril-base-cased  | ಅಡಿ     |       2 |    0.64 |                         0.9  |                      0.98 |
| google/muril-base-cased  | ಗುಂಡಿ     |       1 |    0.8  |                         0.97 |                      0.92 |
| google/muril-base-cased  | ಗುಂಡಿ     |       2 |    0.2  |                         0.75 |                      0.9  |
| google/muril-base-cased  | ಮಂಡಿ     |       1 |    0.6  |                         1    |                      0.92 |
| google/muril-base-cased  | ಮಂಡಿ     |       2 |    0.4  |                         0.89 |                      1    |
| google/muril-base-cased  | ಮತ     |       1 |    0.96 |                         1    |                      0.93 |
| google/muril-base-cased  | ಮತ     |       2 |    0.04 |                         0.36 |                      1    |
| google/muril-base-cased  | ಮುತ್ತು    |       2 |    0.28 |                         0.81 |                      0.79 |
| google/muril-base-cased  | ಮುತ್ತು    |       1 |    0.72 |                         0.92 |                      0.93 |
| pierluigic/xl-lexeme     | ಅಡಿ     |       1 |    0.36 |                         0.97 |                      0.81 |
| pierluigic/xl-lexeme     | ಅಡಿ     |       2 |    0.64 |                         0.9  |                      0.98 |
| pierluigic/xl-lexeme     | ಗುಂಡಿ     |       1 |    0.8  |                         0.97 |                      0.92 |
| pierluigic/xl-lexeme     | ಗುಂಡಿ     |       2 |    0.2  |                         0.75 |                      0.9  |
| pierluigic/xl-lexeme     | ಮಂಡಿ     |       1 |    0.6  |                         1    |                      0.92 |
| pierluigic/xl-lexeme     | ಮಂಡಿ     |       2 |    0.4  |                         0.89 |                      1    |
| pierluigic/xl-lexeme     | ಮತ     |       1 |    0.96 |                         1    |                      0.93 |
| pierluigic/xl-lexeme     | ಮತ     |       2 |    0.04 |                         0.36 |                      1    |
| pierluigic/xl-lexeme     | ಮುತ್ತು    |       2 |    0.28 |                         0.81 |                      0.79 |
| pierluigic/xl-lexeme     | ಮುತ್ತು    |       1 |    0.72 |                         0.92 |                      0.93 |


### Table 3: Combined Summary for `predicted_sense` and `llm_annotation`

| Model                    | Word   |   Sense |   Prior |   Precision (predicted_sense) |   Recall (predicted_sense) |   Precision (llm_annotation) |   Recall (llm_annotation) |
|:-------------------------|:-------|--------:|--------:|------------------------------:|---------------------------:|-----------------------------:|--------------------------:|
| l3cube-pune/kannada-bert | ಅಡಿ     |       1 |    0.36 |                          0.97 |                       0.89 |                         0.97 |                      0.81 |
| l3cube-pune/kannada-bert | ಅಡಿ     |       2 |    0.64 |                          0.94 |                       0.98 |                         0.9  |                      0.98 |
| l3cube-pune/kannada-bert | ಗುಂಡಿ     |       1 |    0.8  |                          0.85 |                       0.98 |                         0.97 |                      0.92 |
| l3cube-pune/kannada-bert | ಗುಂಡಿ     |       2 |    0.2  |                          0.75 |                       0.3  |                         0.75 |                      0.9  |
| l3cube-pune/kannada-bert | ಮಂಡಿ     |       1 |    0.6  |                          1    |                       0.88 |                         1    |                      0.92 |
| l3cube-pune/kannada-bert | ಮಂಡಿ     |       2 |    0.4  |                          0.85 |                       1    |                         0.89 |                      1    |
| l3cube-pune/kannada-bert | ಮತ     |       1 |    0.96 |                          1    |                       0.86 |                         1    |                      0.93 |
| l3cube-pune/kannada-bert | ಮತ     |       2 |    0.04 |                          0.24 |                       1    |                         0.36 |                      1    |
| l3cube-pune/kannada-bert | ಮುತ್ತು    |       2 |    0.28 |                          0.41 |                       1    |                         0.81 |                      0.79 |
| l3cube-pune/kannada-bert | ಮುತ್ತು    |       1 |    0.72 |                          1    |                       0.43 |                         0.92 |                      0.93 |
| google/muril-base-cased  | ಅಡಿ     |       1 |    0.36 |                          0.97 |                       0.86 |                         0.97 |                      0.81 |
| google/muril-base-cased  | ಅಡಿ     |       2 |    0.64 |                          0.93 |                       0.98 |                         0.9  |                      0.98 |
| google/muril-base-cased  | ಗುಂಡಿ     |       1 |    0.8  |                          0.83 |                       0.99 |                         0.97 |                      0.92 |
| google/muril-base-cased  | ಗುಂಡಿ     |       2 |    0.2  |                          0.8  |                       0.2  |                         0.75 |                      0.9  |
| google/muril-base-cased  | ಮಂಡಿ     |       1 |    0.6  |                          1    |                       0.88 |                         1    |                      0.92 |
| google/muril-base-cased  | ಮಂಡಿ     |       2 |    0.4  |                          0.85 |                       1    |                         0.89 |                      1    |
| google/muril-base-cased  | ಮತ     |       1 |    0.96 |                          1    |                       0.85 |                         1    |                      0.93 |
| google/muril-base-cased  | ಮತ     |       2 |    0.04 |                          0.22 |                       1    |                         0.36 |                      1    |
| google/muril-base-cased  | ಮುತ್ತು    |       2 |    0.28 |                          0.41 |                       1    |                         0.81 |                      0.79 |
| google/muril-base-cased  | ಮುತ್ತು    |       1 |    0.72 |                          1    |                       0.43 |                         0.92 |                      0.93 |
| pierluigic/xl-lexeme     | ಅಡಿ     |       1 |    0.36 |                          0.93 |                       0.75 |                         0.97 |                      0.81 |
| pierluigic/xl-lexeme     | ಅಡಿ     |       2 |    0.64 |                          0.87 |                       0.97 |                         0.9  |                      0.98 |
| pierluigic/xl-lexeme     | ಗುಂಡಿ     |       1 |    0.8  |                          0.79 |                       0.88 |                         0.97 |                      0.92 |
| pierluigic/xl-lexeme     | ಗುಂಡಿ     |       2 |    0.2  |                          0.09 |                       0.05 |                         0.75 |                      0.9  |
| pierluigic/xl-lexeme     | ಮಂಡಿ     |       1 |    0.6  |                          0.97 |                       0.57 |                         1    |                      0.92 |
| pierluigic/xl-lexeme     | ಮಂಡಿ     |       2 |    0.4  |                          0.6  |                       0.98 |                         0.89 |                      1    |
| pierluigic/xl-lexeme     | ಮತ     |       1 |    0.96 |                          0.98 |                       0.9  |                         1    |                      0.93 |
| pierluigic/xl-lexeme     | ಮತ     |       2 |    0.04 |                          0.17 |                       0.5  |                         0.36 |                      1    |
| pierluigic/xl-lexeme     | ಮುತ್ತು    |       2 |    0.28 |                          0.09 |                       0.07 |                         0.81 |                      0.79 |
| pierluigic/xl-lexeme     | ಮುತ್ತು    |       1 |    0.72 |                          0.67 |                       0.72 |                         0.92 |                      0.93 |


