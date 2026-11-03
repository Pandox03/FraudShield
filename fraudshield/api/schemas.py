from pydantic import BaseModel, ConfigDict, Field

from fraudshield.config import PCA_FEATURES


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Time: float = Field(ge=0, description="Secondes depuis la 1re transaction")
    Amount: float = Field(ge=0, le=30000, description="Montant de la transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


# vérifier que le schéma couvre bien toutes les features PCA
assert [f"V{i}" for i in range(1, 29)] == PCA_FEATURES


class PredictionOut(BaseModel):
    probabilite_fraude: float = Field(ge=0, le=1)
    alerte: bool
    seuil: float
    model_version: str
    modele: str
