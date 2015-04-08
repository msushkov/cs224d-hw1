rm -f assignment1.zip 
zip -r assignment1.zip . -x "*.git*" "*cs224d/datasets*" "*.ipynb_checkpoints*" "*README.md" "*collectSubmission.sh" "*requirements.txt" \
	"saved_params_10000.npy" "saved_params_20000.npy" "saved_params_30000.npy" "saved_params_40000.npy" "saved_params_50000.npy" \
	"saved_params_60000.npy" "saved_params_70000.npy" "saved_params_80000.npy" "saved_params_90000.npy" "saved_params_100000.npy" \
	"saved_params_110000.npy" "saved_params_120000.npy" "saved_params_130000.npy" "saved_params_140000.npy" "saved_params_150000.npy" \
	"saved_params_160000.npy" "saved_params_170000.npy" "saved_params_180000.npy" "saved_params_190000.npy"