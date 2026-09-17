from pydantic import BaseModel, Field, EmailStr

class StudentValidator(BaseModel):
    nume: str = Field(min_length=2, max_length=50)
    varsta: int = Field(gt=0)
    email: EmailStr