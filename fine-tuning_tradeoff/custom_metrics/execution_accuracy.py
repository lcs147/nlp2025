from common import * # import all file paths and common functions and variables

import sqlite3
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
import pandas as pd
import os

class ExecutionAccuracy(BaseMetric):
    def __init__(self, threshold: float = 1):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase) -> float:
        db_id = test_case.context[0]
        db_path = f"{PATH_SPIDER_FULL}/database/{db_id}/{db_id}.sqlite"

        actual_result = self._execute_sql_query(db_path, test_case.actual_output)
        expected_result = self._execute_sql_query(db_path, test_case.expected_output)

        if actual_result["success"] and expected_result["success"]:
            if set(list(actual_result["data"])) == set(list(expected_result["data"])):
                self.score = 1.0
                self.reason = "Correct result."
            else:
                self.score = 0.0
                self.reason = "Incorrect result."
        else:
            self.score = 0.0
            if not actual_result["success"]:
                self.reason = f"Generated SQL execution failed: {actual_result['error']}"
            elif not expected_result["success"]:
                self.reason = f"Ground truth SQL execution failed: {expected_result['error']}"
            else:
                self.reason = "Unknown error during SQL execution."

        self.success = (self.score >= self.threshold)
        return self.score
    
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)
    
    
    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Execution Accuracy"

    def _execute_sql_query(self, db_path, query):
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            data = cursor.fetchall()
            return {"success": True, "data": data, "error": None}
        except sqlite3.Error as e:
            return {"success": False, "data": None, "error": str(e)}
        finally:
            if conn:
                conn.close()