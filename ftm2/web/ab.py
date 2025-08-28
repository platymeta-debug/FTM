# [ANCHOR:WEB_AB]
import json, copy
from fastapi import APIRouter, Depends, Body
from ftm2.web.auth import verify
from ftm2.abtest.run_ab import apply_overrides, run_variant
from ftm2.config.settings import cfg as base_cfg
router = APIRouter(prefix="/ab", tags=["ab"])

@router.post("/run")
def run(payload: dict = Body(...), _: None = Depends(verify)):
    A = payload.get("A") or {}
    B = payload.get("B") or {}
    cfgA = apply_overrides(copy.deepcopy(base_cfg), A)
    cfgB = apply_overrides(copy.deepcopy(base_cfg), B)
    resA = run_variant(cfgA); resB = run_variant(cfgB)
    return {"A":resA, "B":resB, "diff":{
        "expectancy": resB["expectancy"]-resA["expectancy"],
        "pf": resB["pf"]-resA["pf"],
        "sharpe": resB["sharpe"]-resA["sharpe"]
    }}
