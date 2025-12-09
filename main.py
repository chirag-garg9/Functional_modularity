# main.py
import glob
from train import main
from tqdm import tqdm


configs = sorted(glob.glob("configs/*.yaml"))
for cfg in tqdm(configs):
    print(f"\n🚀 Running experiment: {cfg}")
    main(cfg)

