# KL test
python src/train_ldm.py logger.wandb.name=kl_1e-1 ldm_module.vae_ckpt_path=ckpts/vae/kl_1e-1/model.ckpt
python src/train_ldm.py logger.wandb.name=kl_1e-2 ldm_module.vae_ckpt_path=ckpts/vae/kl_1e-2/model.ckpt
python src/train_ldm.py logger.wandb.name=kl_1e-3 ldm_module.vae_ckpt_path=ckpts/vae/kl_1e-3/model.ckpt
python src/train_ldm.py logger.wandb.name=kl_1e-4 ldm_module.vae_ckpt_path=ckpts/vae/kl_1e-4/model.ckpt
python src/train_ldm.py logger.wandb.name=kl_1e-5 ldm_module.vae_ckpt_path=ckpts/vae/kl_1e-5/model.ckpt

# Noise 
python src/train_ldm.py logger.wandb.name=kl_1e-5_noise ldm_module.vae_ckpt_path=ckpts/vae/kl_1e-5_noise/model.ckpt

# Foundation alignment 
python src/train_ldm.py logger.wandb.name=kl_1e-5_fa_1e-1 ldm_module.vae_ckpt_path=ckpts/vae/kl_1e-5_fa_1e-1/model.ckpt
python src/train_ldm.py logger.wandb.name=kl_1e-5_fa_1e-2 ldm_module.vae_ckpt_path=ckpts/vae/kl_1e-5_fa_1e-2/model.ckpt
python src/train_ldm.py logger.wandb.name=kl_1e-5_fa_1e-3 ldm_module.vae_ckpt_path=ckpts/vae/kl_1e-5_fa_1e-3/model.ckpt

# EMA 
python src/train_ldm.py logger.wandb.name=kl_1e-5_ema ldm_module.vae_ckpt_path=ckpts/vae/kl_1e-5/model.ckpt +callbacks.ema=ema

# mp_120
python src/train_ldm.py experiment=mp_120/ldm_null
python src/train_ldm.py experiment=mp_120/ldm_chemical_system_finetuned_from_rl_dng
python src/train_ldm.py experiment=mp_120/ldm_composition_finetuned_from_rl_dng
python src/train_ldm.py experiment=mp_120/ldm_composition_lora_finetuned_from_rl_dng

# mp_amorphous
python src/train_ldm.py experiment=mp_amorphous/mp_amorphous

# alex_mp_20_bandgap
python src/train_ldm.py experiment=alex_mp_20_bandgap/ldm_bandgap
python src/train_ldm.py experiment=alex_mp_20_bandgap/ldm_bandgap_lora