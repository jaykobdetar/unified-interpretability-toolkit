"""Pure DATA source/build binding controls; no browser, model or source writes."""
from pathlib import Path
import unittest
from unittest.mock import patch
import data_browser_driver as driver


class SourceBinding(unittest.TestCase):
    def validate(self, changed='', actual_sha='b'*64, expected_sha='b'*64):
        commit='a'*40
        outputs=[commit,changed,'c'*40,'d'*40]
        with patch.object(driver.subprocess,'check_output',side_effect=outputs),patch.object(Path,'is_file',return_value=True),patch.object(driver,'sha',return_value=actual_sha):
            return driver.validate_source_binding(Path('/synthetic'),Path('/synthetic/binary'),commit,expected_sha)

    def test_exact_selected_source_and_binary(self):
        value=self.validate()
        self.assertEqual(value['production_commit'],'a'*40)
        self.assertTrue(value['binary_pin_verified'])
        self.assertEqual(value['historical_candidate'],driver.CANDIDATE)

    def test_changed_production_is_rejected(self):
        for path in ['src/main.rs','web/app.js','web/style.css']:
            with self.subTest(path=path),self.assertRaisesRegex(AssertionError,'Production sources differ'):
                self.validate(changed=path+'\n')

    def test_changed_or_missing_binary_pin_is_rejected(self):
        with self.assertRaisesRegex(AssertionError,'Binary differs'):
            self.validate(actual_sha='e'*64)
        with self.assertRaisesRegex(AssertionError,'requires its recorded binary'):
            self.validate(expected_sha=None)

    def test_mutable_or_missing_source_is_rejected(self):
        with self.assertRaisesRegex(AssertionError,'immutable source'):
            driver.validate_source_binding(Path('/synthetic'),Path('/synthetic/binary'),'HEAD','b'*64)
        with patch.object(driver.subprocess,'check_output',return_value='e'*40),self.assertRaisesRegex(AssertionError,'resolve exactly'):
            driver.validate_source_binding(Path('/synthetic'),Path('/synthetic/binary'),'a'*40,'b'*64)


if __name__=='__main__':unittest.main()
