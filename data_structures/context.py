from dataclasses import dataclass

from models.loss import FMOLoss
from models.rendering import RenderingKaolin


@dataclass
class ContextManager:

    renderer: RenderingKaolin
    loss_function: FMOLoss
