import unittest

from salsbury_md_analysis.validation import integer_at_least, positive_integer


class CustomValidationError(ValueError):
    pass


class ValidationTests(unittest.TestCase):
    def test_positive_integer_rejects_boolean_zero_and_float(self):
        for value in (True, 0, -1, 1.0):
            with self.subTest(value=value), self.assertRaises(CustomValidationError):
                positive_integer(value, "count", error_type=CustomValidationError)
        self.assertEqual(
            positive_integer(2, "count", error_type=CustomValidationError), 2
        )

    def test_integer_at_least_preserves_declared_bound_and_error_type(self):
        with self.assertRaisesRegex(CustomValidationError, "at least 24"):
            integer_at_least(23, "points", 24, error_type=CustomValidationError)
        self.assertEqual(
            integer_at_least(24, "points", 24, error_type=CustomValidationError),
            24,
        )

    def test_validator_definition_rejects_invalid_minimum(self):
        with self.assertRaises(ValueError):
            integer_at_least(3, "count", 0)


if __name__ == "__main__":
    unittest.main()
