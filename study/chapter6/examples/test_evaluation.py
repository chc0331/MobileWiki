"""Small regression checks for the textbook's evaluation denominators and data split."""
import copy
import unittest

from run_classification import LABELS, OUTSIDE, check_data, metrics, predicted_rows, toy_report


def row(label, probabilities, group='one'):
    return dict(label=label, probabilities=probabilities, group_id=group)


class EvaluationChecks(unittest.TestCase):
    def test_textbook_matrix(self):
        report = toy_report()['metrics']
        self.assertEqual(report['matrix'], [[3, 1, 0, 1], [1, 2, 1, 0], [0, 0, 2, 1], [1, 1, 0, 2]])
        self.assertAlmostEqual(report['per_class']['IT']['precision'], 3 / 5)
        self.assertAlmostEqual(report['per_class']['IT']['recall'], 3 / 5)
        self.assertAlmostEqual(report['macro_f1'], (0.6 + 0.5 + 2 / 3) / 3)
        self.assertAlmostEqual(report['automatic_error_rate'], 5 / 12)
        self.assertAlmostEqual(report['hold_rate'], 4 / 16)
        self.assertAlmostEqual(report['known_accuracy'], 7 / 12)

    def test_held_positive_stays_in_recall(self):
        m = metrics([row('IT', [.8, .1, .1]), row('IT', [.4, .3, .3])], .6)
        self.assertEqual(m['per_class']['IT']['precision'], 1)
        self.assertEqual(m['per_class']['IT']['recall'], .5)
        self.assertEqual(m['known_accuracy'], .5)

    def test_outside_false_accept_is_false_positive(self):
        m = metrics([row('IT', [.8, .1, .1]), row(OUTSIDE, [.9, .05, .05])], .6)
        self.assertEqual(m['per_class']['IT']['precision'], .5)
        self.assertEqual(m['automatic_error_rate'], .5)
        self.assertEqual(m['outside_false_accept_rate'], 1)

    def test_all_held_is_not_perfect_automatic_accuracy(self):
        m = toy_report()['all_held']
        self.assertIsNone(m['automatic_error_rate'])
        self.assertEqual(m['hold_rate'], 1)
        self.assertEqual(m['known_accuracy'], 0)
        self.assertTrue(m['per_class']['IT']['precision_undefined'])

    def test_boundary_and_tie_follow_class_order(self):
        m = metrics([row('경제', [.5, .5, 0])], .5)
        self.assertEqual(m['matrix'][1][0], 1)
        self.assertEqual(metrics([row('IT', [.5, .3, .2])], .5001)['automatic_count'], 0)

    def test_no_supported_classes_is_marked(self):
        m = metrics([row(OUTSIDE, [.4, .3, .3])], .6)
        self.assertIsNone(m['known_accuracy'])
        self.assertTrue(all(c['recall_undefined'] for c in m['per_class'].values()))

    def test_metrics_match_sklearn(self):
        from sklearn.metrics import precision_recall_fscore_support
        matrix = toy_report()['metrics']['matrix']
        truths, predictions = [], []
        for i, counts in enumerate(matrix):
            for j, count in enumerate(counts):
                truths.extend([(LABELS + [OUTSIDE])[i]] * count)
                predictions.extend([(LABELS + ['보류'])[j]] * count)
        p, r, f, support = precision_recall_fscore_support(truths, predictions, labels=LABELS, zero_division=0)
        for i, label in enumerate(LABELS):
            m = toy_report()['metrics']['per_class'][label]
            self.assertAlmostEqual(m['precision'], p[i])
            self.assertAlmostEqual(m['recall'], r[i])
            self.assertAlmostEqual(m['f1'], f[i])
            self.assertEqual(m['support'], support[i])

    def test_input_ranges_describe_the_input_actually_used(self):
        document = dict(doc_id='long-en', group_id='long', language='en', label='IT', clean_text='x' * 30)
        detail = dict(full_token_count=200, length_bucket='long', chunk_count=2,
                      source_ranges=[[0, 12], [14, 30]], first_source_range=[0, 18])
        full = predicted_rows([document], [[.8, .1, .1]], [detail], 'full_text')[0]
        first = predicted_rows([document], [[.8, .1, .1]], [detail], 'first_window')[0]
        self.assertEqual(full['source_ranges'], [[0, 30]])
        self.assertEqual(first['source_ranges'], [[0, 18]])
        self.assertEqual(first['chunk_count'], 1)
        self.assertEqual(first['full_token_count'], 200)


class DataChecks(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(doc_id=f'{split}-{label}', group_id=f'{split}-{label}', language='ko',
                          label=label, split=split, clean_text=f'{split} unique {label}')
                     for split in ('train', 'validation', 'final_test') for label in LABELS]

    def test_valid_split(self):
        self.assertEqual(check_data(self.rows)['train']['documents'], 3)

    def test_translation_must_stay_with_original(self):
        translated = copy.deepcopy(self.rows[0])
        translated.update(doc_id='translation', split='validation', language='en', clean_text='new translated text')
        with self.assertRaisesRegex(ValueError, '분할 또는 라벨'):
            check_data(self.rows + [translated])

    def test_duplicate_text_and_empty_input(self):
        bad = copy.deepcopy(self.rows)
        bad[1]['clean_text'] = bad[0]['clean_text']
        with self.assertRaisesRegex(ValueError, '완전히 같은 본문'):
            check_data(bad)
        bad[1]['clean_text'] = '  '
        with self.assertRaisesRegex(ValueError, '빈 본문'):
            check_data(bad)

    def test_outside_does_not_become_a_fourth_training_class(self):
        bad = copy.deepcopy(self.rows)
        bad[0]['label'] = OUTSIDE
        with self.assertRaisesRegex(ValueError, '범위 밖'):
            check_data(bad)


if __name__ == '__main__':
    unittest.main()
