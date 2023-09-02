# Before executing the program
# pip install presidio-analyzer
# pip install presidio-anonymizer
# python -m spacy download en_core_web_lg

import pprint

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


class Anonymize():
    def __init__(self):
        self.module = "Anonymizer"

    def anlalyze_text(self, text):
        """
        This module identifies the existence of PII data in the text based on the
        provided entities
        :param text: The text to analyze
        :return: Results from the analyze which will be fed to the anonymizer
        """
        pii_analyzer = AnalyzerEngine()
        results = pii_analyzer.analyze(text=text,
                                       entities=["PHONE_NUMBER", "EMAIL_ADDRESS"],
                                       language='en',return_decision_process=True)

        decision_process = results[0].analysis_explanation
        pp = pprint.PrettyPrinter()
        print("Decision process output:\n")
        pp.pprint(decision_process.__dict__)

        return results

    def anonymize_text(self, text):
        """
        This module does the actual anonymization
        :param text: The text to anonymze
        :return: Anonymed text
        """
        results = self.anlalyze_text(text)
        anonymizer = AnonymizerEngine()

        # Define anonymization operators
        operators = {
            "DEFAULT": OperatorConfig("redact",{}),
            "PHONE_NUMBER": OperatorConfig(
                "mask",
                {
                    "type": "mask",
                    "masking_char": "*",
                    "chars_to_mask": 12,
                    "from_end": True,
                },
            ),
            "EMAIL_ADDRESS": OperatorConfig("replace" ,{"new_value": "{EMAIL}"})
        }

        anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results,operators=operators)
        return anonymized_text


if __name__ == "__main__":
    an = Anonymize()
    text = "Rajib can be reached at 3456789055, his email is rajib@abc.com. His lucky number is 4567896789"
    anonymize_text = an.anonymize_text(text)
    print(anonymize_text.text)
