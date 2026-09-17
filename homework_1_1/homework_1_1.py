import json

def read_student():
    with open("homework_1_1/student.json", "r") as file:
        student_data = json.load(file)

    return student_data