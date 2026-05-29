import sys
from pathlib import Path
SCRIPT_DIR = Path(".").resolve()
sys.path.insert(0, str(SCRIPT_DIR))
from single_task.train_ppo import Args
from single_task.utils import get_env
args = Args(mimic_task="TwinkleTwinkleRousseau")
try:
    env = get_env(args)
    print("Has task directly:", hasattr(env, "task"))
    print("Has env.task:", hasattr(env.env, "task"))
except Exception as e:
    print(e)
