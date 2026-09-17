from pydantic import ValidationError
import pytest
from homework_1_1.student_validator import StudentValidator
from homework_1_1.homework_1_1 import read_student

json_student_data = read_student()

def test_student_valid():
    student = StudentValidator(**json_student_data)

    assert student.nume == "Ana Popescu"
    assert student.varsta == 20
    assert student.email == "ana.popescu@example.com"

def test_student_invalid():
    invalid_data = {
        **json_student_data,
        "nume": "",
        "varsta": 9999,
        "email": "",
    }

    with pytest.raises(ValidationError):
        StudentValidator(**invalid_data)


def test_name_valid():
    student = StudentValidator(**json_student_data)

    assert student.nume == "Ana Popescu"


def test_name_invalid():
    invalid_data = {
        **json_student_data,
        "nume": ""
    }

    with pytest.raises(ValidationError):
        StudentValidator(**invalid_data)

def test_age_valid():
    student = StudentValidator(**json_student_data)

    assert student.varsta == 20

def test_age_invalid():
    invalid_data = {
        **json_student_data,
        "varsta": 0,
    }

    with pytest.raises(ValidationError):
        StudentValidator(**invalid_data)

def test_email_valid():
    student = StudentValidator(**json_student_data)

    assert student.email == "ana.popescu@example.com"

def test_email_invalid():
    invalid_data = {
        **json_student_data,
        "email": "test",
    }

    with pytest.raises(ValidationError):
        StudentValidator(**invalid_data)
