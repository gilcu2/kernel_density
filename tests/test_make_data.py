import unittest
import sample


class TestMake_data(unittest.TestCase):

    def test_make_data(self):
        data = sample.make_data(10)
        self.assertEqual(data.size, 10)


if __name__ == '__main__':
    unittest.main()
