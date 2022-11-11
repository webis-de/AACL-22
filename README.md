On the Perceptibility of Differences in Stance Bias in Argumentation
==============================

Most relevant natural language processing research treats bias as an absolute concept: a text is classified as biased or not with respect to some criterion. Each text is thus considered in isolation, while the decision as to whether a particular text has crossed the proverbial line between biased and non-biased is highly subjective. In this paper, bias is treated for the first time as a  relative concept by asking questions such as "Is a text $X$ more [less, equal] biased than a text $Y$?'' In such a model, bias becomes a kind of preference relation that induces a partial ordering from least biased to most biased texts, without requiring a final decision on where to draw the line. A prerequisite for this treatment of bias is the ability of human subjects to perceive relative bias differences in the first place. Therefore, we selected a specific type of bias in argumentation, the stance bias, and carefully designed a crowdsourcing study showing that differences in stance bias are perceptible when (light) support is provided through training or visual aid.

## Structure

```bash
├── README.md
├── analysis    
├── data
│   ├── crowdsourced
│   └── external
│       └── bias_related_lexicons
├── experiments
│   ├── cs1
│   │   ├── surveys
│   │   │   └── stance_surveys
│   │   └── templates
│   │       └── stance_templates
│   └── cs2
│       ├── surveys
│       │   └── stance_surveys
│       └── templates
│           └── stance_templates
├── utils
```

### Instructions

1. Generate the HIT templates with `stance_survey_generator` file in `experiments` directory.

2. For running the analysis use the Jupyer notebooks in the `analysis` directory ([Table 1c](https://github.com/webis-de/aacl-ijcnlp-22-differential-bias/blob/main/analysis/labeling_accuracies.ipynb), [Table 2a](https://github.com/webis-de/aacl-ijcnlp-22-differential-bias/blob/main/analysis/setups.ipynb) , [Figure 2b](https://github.com/webis-de/aacl-ijcnlp-22-differential-bias/blob/main/analysis/setups.ipynb), [Table 2c](https://github.com/webis-de/aacl-ijcnlp-22-differential-bias/blob/main/analysis/accs_analysis.ipynb), [Table 3](https://github.com/webis-de/aacl-ijcnlp-22-differential-bias/analysis/competence_filtering.ipynb), [Lexical strength analysis](https://github.com/webis-de/aacl-ijcnlp-22-differential-bias/blob/main/analysis/lex_strength_analysis.ipynb))

### Reference

On the Perceptibility of Differences in Stance Bias in Argumentation can be found at [TBA](tba)

```bash
TBA
```