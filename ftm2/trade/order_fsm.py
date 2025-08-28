# [ANCHOR:ORDER_FSM]
from enum import Enum, auto

class OState(Enum):
    IDLE = auto()
    NEW = auto()
    PARTIAL = auto()
    FILLED = auto()
    BRACKETS_SET = auto()
    CANCELLED = auto()
    FAILED = auto()


class OFSM:
    def __init__(self, sym, cid):
        self.sym = sym
        self.cid = cid
        self.state = OState.IDLE
        self.filled_qty = 0.0

    def to(self, st):
        self.state = st
        return self
