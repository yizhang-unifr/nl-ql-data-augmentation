
from gpt3_prompt import *
import unittest
### Test cases ###


class TestGPT3Prompt(unittest.TestCase):
    def test_prompt_builder(self):
        d = {"db_id": "department_management",
             "generated_ast": "Root1(3) Root(3) Sel(0) N(0) A(3) Op(0) C(0) T(1) Filter(4) A(0) Op(0) C(9) T(1) V(0)",
             "generated_query": "SELECT COUNT(*) FROM head AS T1 WHERE T1.age < 56",
             "generated_values": [
                 "56"
             ],
             "original_AST": "Root1(3) Root(3) Sel(0) N(0) A(3) Op(0) C(0) T(1) Filter(5) A(0) Op(0) C(9) T(1) V(0)",
             "original_query": "SELECT COUNT(*) FROM head AS T1 WHERE T1.age > 56",
             "original_values": [
                 "56"
             ],
             "question": "How many heads of the departments are older than 56 ?"}
        res = "# Translate the following SQL query into natural language question:\n" +\
              "#\n" + \
              "# SQL query: SELECT COUNT(*) FROM head AS T1 WHERE T1.age > 56\n" + \
              "# Natural language question: How many heads of the departments are older than 56 ?\n" + \
              "#\n" + \
              "# SQL query: SELECT COUNT(*) FROM head AS T1 WHERE T1.age < 56\n" + \
              "# Natural language question: "
        # print(res)
        # print(prompt_builder(d))
        self.assertEqual(prompt_builder(d), res)
