import unittest
from src.train import (
    main
)
from src.recipes.clarity_recipe.evaluate import run_calculate_si
from src.recipes.clarity_recipe.enhance import enhance

class ClaritySanityCheck(unittest.TestCase):
    def test_enhance(self):
        """
        python -m unittest -v test.test_clarity_baseline.ClaritySanityCheck.test_enhance
        """
        enhance()
    def test_evaluate(self):
        """
        python -m unittest -v test.test_train.ClaritySanityCheck.test_evaluate
        """
        run_calculate_si