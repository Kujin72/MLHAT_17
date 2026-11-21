import tempfile
import unittest
from pathlib import Path

from mlhat_rebuild.benchmark_data import read_benchmark_arff, verify_paper_shape


class BenchmarkTests(unittest.TestCase):
    def _make_arff(self, folder, text):
        path = Path(folder) / 'tiny.arff'
        path.write_text(text, encoding='utf-8')
        return path

    def test_meka_last_labels_one_hot(self):
        with tempfile.TemporaryDirectory() as folder:
            p = self._make_arff(folder, '''@relation 'tiny: -C -2'
@attribute score NUMERIC
@attribute color {red,blue}
@attribute label1 {0,1}
@attribute label2 {0,1}
@data
0.5,red,1,0
0.7,blue,0,1
''')
            data = read_benchmark_arff(p, 'Flags')
            self.assertEqual(data.records[0], ((0.5, 1.0, 0.0), (1, 0)))
            self.assertEqual(data.encoded_features, 3)
            self.assertEqual(data.raw_features, 2)
            ok, differences = verify_paper_shape(data)
            self.assertFalse(ok)
            self.assertTrue(differences)

    def test_meka_first_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            p = self._make_arff(folder, '''@relation "tiny: -C 2"
@attribute label1 {0,1}
@attribute label2 {0,1}
@attribute x NUMERIC
@attribute z NUMERIC
@data
1,0,1.5,2.5
''')
            data = read_benchmark_arff(p, 'Flags')
            self.assertEqual(data.records[0], ((1.5, 2.5), (1, 0)))

    def test_mulan_xml_nonlast_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            p = self._make_arff(folder, '''@relation toy
@attribute l1 {0,1}
@attribute age NUMERIC
@attribute l2 {0,1}
@data
1,2,0
''')
            xml = Path(folder) / 'labels.xml'
            xml.write_text('<labels><label name="l2"/><label name="l1"/></labels>', encoding='utf8')
            data = read_benchmark_arff(p, 'Flags', xml)
            self.assertEqual(data.records[0], ((2.0,), (0, 1)))
            self.assertEqual(data.label_names, ('l2', 'l1'))

    def test_missing_feature_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = self._make_arff(folder, '''@relation 'toy: -C -1'
@attribute x NUMERIC
@attribute l {0,1}
@data
?,1
''')
            with self.assertRaisesRegex(ValueError, 'Missing feature'):
                read_benchmark_arff(p, 'Flags')

    def test_label_definition_required(self):
        with tempfile.TemporaryDirectory() as folder:
            p = self._make_arff(folder, '''@relation toy
@attribute x NUMERIC
@attribute l {0,1}
@data
1,1
''')
            with self.assertRaisesRegex(ValueError, 'Label definition absent'):
                read_benchmark_arff(p, 'Flags')

if __name__ == '__main__': unittest.main()
