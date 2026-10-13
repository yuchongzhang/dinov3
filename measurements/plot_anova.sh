#!/bin/bash

source /home/yucz/projects/aip-papyan/yucz/venvs/dinov3_env/bin/activate

python -m measurements.plot_anova --inputs measurements/anova_results_no_cls/vits_claude_anova.json measurements/anova_results_no_cls/vits_1L_claude_anova.json measurements/anova_results_no_cls/vits_3L_claude_anova.json measurements/anova_results_no_cls/vits_5L_claude_anova.json measurements/anova_results_no_cls/vits_6L_claude_anova.json --output measurements/anova_results_no_cls/anova_figure.png