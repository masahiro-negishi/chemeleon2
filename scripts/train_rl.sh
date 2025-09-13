# DNG
# python src/train_rl.py rl_module.reward_fn.reward_type=dng logger.wandb.name=dng_group rl_module.rl_configs.num_group_samples=64 rl_module.rl_configs.group_reward_norm=true data.batch_size=5
python src/train_rl.py experiment=rl_dng

# bandgap
python src/train_rl.py experiment=alex_mp_20_bandgap/rl_bandgap