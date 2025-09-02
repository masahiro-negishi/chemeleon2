# DNG (mp-20)
# python src/evaluate.py --structure_path=benchmarks/dng/adit_dng_mp_20.json.gz --output_file=benchmarks/dng/results/mp_20/adit_dng_mp_20_ref_mp_20.csv
# python src/evaluate.py --structure_path=benchmarks/dng/chemeleon_dng_mp_20.json.gz --output_file=benchmarks/dng/results/mp_20/chemeleon_dng_mp_20_ref_mp_20.csv
# python src/evaluate.py --structure_path=benchmarks/dng/diffcsp_dng_mp_20.json.gz --output_file=benchmarks/dng/results/mp_20/diffcsp_dng_mp_20_ref_mp_20.csv
# python src/evaluate.py --structure_path=benchmarks/dng/chemeleon2_rl_dng_mp_20.json.gz --output_file=benchmarks/dng/results/mp_20/chemeleon2_rl_dng_mp_20_ref_mp_20.csv
# python src/evaluate.py --structure_path=benchmarks/dng/chemeleon2_ldm_null_mp_20.json.gz --output_file=benchmarks/dng/results/mp_20/chemeleon2_ldm_null_mp_20_ref_mp_20.csv
# python src/evaluate.py --structure_path=benchmarks/dng/mattergen_dng_mp_20.json.gz --output_file=benchmarks/dng/results/mp_20/mattergen_dng_mp_20_ref_mp_20.csv

# DNG (alex-mp-20)
python -u src/evaluate.py --structure_path=benchmarks/dng/chemeleon_dng_alex_mp_20.json.gz --output_file=benchmarks/dng/results/alex_mp_20/chemeleon_dng_alex_mp_20_ref_alex_mp_20.csv --reference_dataset=alex-mp-20 --phase_diagram=alex-mp-20
python -u src/evaluate.py --structure_path=benchmarks/dng/mattergen_dng_alex_mp_20.json.gz --output_file=benchmarks/dng/results/alex_mp_20/mattergen_dng_alex_mp_20_ref_alex_mp_20.csv --reference_dataset=alex-mp-20 --phase_diagram=alex-mp-20
python -u src/evaluate.py --structure_path=benchmarks/dng/chemeleon2_ldm_null_alex_mp_20.json.gz --output_file=benchmarks/dng/results/alex_mp_20/chemeleon2_ldm_null_alex_mp_20_ref_alex_mp_20.csv --reference_dataset=alex-mp-20 --phase_diagram=alex-mp-20
python -u src/evaluate.py --structure_path=benchmarks/dng/chemeleon2_rl_dng_alex_mp_20.json.gz --output_file=benchmarks/dng/results/alex_mp_20/chemeleon2_rl_dng_alex_mp_20_ref_alex_mp_20.csv --reference_dataset=alex-mp-20 --phase_diagram=alex-mp-20