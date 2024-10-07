Code for some [follow up experiments] to [Berglund *et. al.*, (2023)](https://arxiv.org/abs/2309.00667)

The experiments require a series of steps to run:

0. API key \
Make a .env file with your OpenAI API key as `OPENAI_API_KEY`

1. Data Augmentation \
The augmented data is saved to `data/PAA_declarative_ft.jsonl`. To rerun data augmentation, run `scripts/augment_data.py`

2. Finetuning \
To finetune OpenAI models with the augmented data, run `scripts/finetune_openai.py`

3. Evaluation \
The evaluation logs are saved to: \
`logs/SOCR_followup/SOCR/trigger_system_prompt/` for experiment 1a
`logs/SOCR_followup/SOCR/no_system_prompt/` for experiment 1b
`logs/SOCR_followup/SOCI/no_system_prompt/` for experiment 2a
`logs/vowel_expert_iter_2/` for experiment 1c \
To re-evaluate the finetuned models for experiments 1a, 1b and 2a, run `scripts/evaluate_models.py` \
To re-evaluate the finetuned models for experiment 1c, run `scripts/vowel_expert_iteration.ipynb`

4. Plotting \
To plot the results, run `scripts/plotting.ipynb`

[follow up experiments]: https://docs.google.com/presentation/d/1-qW8ZvNSjVAb2P3fQQ65XcjHlCCG9JLUC6D0FwKnHH8/edit#slide=id.g308ee4e050e_1_42
