import unittest
from protocol import verify_bowl_delivery

class HandoverTests(unittest.TestCase):
    def test_verified_delivery(self):
        verify_bowl_delivery([{'event':'transport_result','passed':True},{'event':'completed'}])

    def test_no_premature_handover(self):
        for events in ([], [{'event':'completed'}], [{'event':'transport_result','passed':True}],
                       [{'event':'transport_result','passed':False},{'event':'completed'}],
                       [{'event':'transport_result','passed':True},{'event':'failed'},{'event':'completed'}],
                       [{'event':'transport_result','passed':True},{'event':'completed'},{'event':'failed'}],
                       [{'event':'completed'},{'event':'transport_result','passed':True}]):
            with self.subTest(events=events), self.assertRaises(ValueError):
                verify_bowl_delivery(events)

if __name__=='__main__':unittest.main()
