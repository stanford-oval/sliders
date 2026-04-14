import os
from datetime import datetime
from sliders.llm.prompts import init_llm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
init_llm(
    prompt_dir=os.path.join(CURRENT_DIR, "prompts"),
    dotenv_path=os.path.join(CURRENT_DIR, "..", ".env"),
    override_env=True,
)


class SlidersGlobal:
    experiment_id = None


SlidersGlobal.experiment_id = datetime.now().strftime("%Y%m%d.%H%M%S.%fZ")
