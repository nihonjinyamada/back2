from pydantic import BaseModel
from typing import Union

class TrainingDataCreate(BaseModel):
    input_text: str
    output_text: Union[str, dict]
    def get_output_text(self):
        if isinstance(self.output_text, dict):
            return json.dumps(self.output_text, ensure_ascii=False)
        return self.output_text

class TrainingDataResponse(BaseModel):
    id: int
    input_text: str
    output_text: str

    class Config:
        orm_mode = True
