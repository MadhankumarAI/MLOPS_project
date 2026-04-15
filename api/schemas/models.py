from pydantic import BaseModel


class NERRequest(BaseModel):
    text: str


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class NERResponse(BaseModel):
    tokens: list[str]
    tags: list[str]
    entities: list[Entity]


class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "es"


class TranslateResponse(BaseModel):
    source_text: str
    translated_text: str
    entities: list[Entity]
    target_lang: str


class StreamMessage(BaseModel):
    type: str
    data: dict
