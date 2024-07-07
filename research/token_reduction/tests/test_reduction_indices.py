import unittest
import torch

from research.token_reduction.layers import (
    TokenDroppingLayer,
    TokenMergingLayer,
    choose_indeces_to_reduce,
    make_available_ids,
)


class TestRandomIndicesOutside(unittest.TestCase):
    def test_basic_case(self):
        batch_size = 2
        seq_len = 7
        result_seq_len = 5
        n_tokens_to_reduce = 2
        indices_to_keep, indices_to_reduct = choose_indeces_to_reduce(
            batch_size, seq_len, result_seq_len, n_tokens_to_reduce
        )
        self.assertEqual(indices_to_keep.numel(), batch_size * result_seq_len)
        self.assertEqual(indices_to_reduct.numel(), batch_size * n_tokens_to_reduce)

    def test_last_token_undroppable(self):
        batch_size = 1
        seq_len = 10
        result_seq_len = 9
        n_tokens_to_reduce = 1
        indices_to_keep, _ = choose_indeces_to_reduce(
            batch_size, seq_len, result_seq_len, n_tokens_to_reduce
        )
        self.assertIn(
            seq_len - 1,
            indices_to_keep.tolist(),
            "Last token index should be in indices_to_keep",
        )

    def test_assertion_error(self):
        batch_size = 1
        seq_len = 5
        result_seq_len = 3
        n_tokens_to_reduce = 3
        with self.assertRaises(AssertionError, msg="Too many tokens to reduce"):
            _ = choose_indeces_to_reduce(
                batch_size, seq_len, result_seq_len, n_tokens_to_reduce
            )

    def test_indices_range(self):
        batch_size = 2
        seq_len = 8
        result_seq_len = 5
        n_tokens_to_reduce = 3
        indices_to_keep, indices_to_reduct = choose_indeces_to_reduce(
            batch_size, seq_len, result_seq_len, n_tokens_to_reduce
        )
        self.assertTrue(torch.all(indices_to_keep < (batch_size * seq_len)))
        self.assertTrue(torch.all(indices_to_reduct < (batch_size * seq_len)))

    def test_all_indices_managed(self):
        batch_size = 3
        seq_len = 6
        result_seq_len = 4
        n_tokens_to_reduce = 2
        indices_to_keep, indices_to_reduct = choose_indeces_to_reduce(
            batch_size, seq_len, result_seq_len, n_tokens_to_reduce
        )
        total_tokens = batch_size * seq_len
        unique_indices = set(indices_to_keep.tolist() + indices_to_reduct.tolist())
        self.assertEqual(
            len(unique_indices),
            total_tokens,
            "All indices should be managed exactly once",
        )

    def test_permutation_property(self):
        batch_size = 2
        seq_len = 9
        result_seq_len = 6
        n_tokens_to_reduce = 2
        indices_to_keep, indices_to_reduct = choose_indeces_to_reduce(
            batch_size, seq_len, result_seq_len, n_tokens_to_reduce
        )
        for i in range(batch_size):
            expected_indices = set(
                range(
                    i * seq_len,
                    i * seq_len + (result_seq_len + n_tokens_to_reduce),
                )
            )
            managed_indices = set(
                indices_to_keep[i * result_seq_len : (i + 1) * result_seq_len].tolist()
                + indices_to_reduct[
                    i * n_tokens_to_reduce : (i + 1) * n_tokens_to_reduce
                ].tolist()
            )
            self.assertEqual(
                expected_indices,
                managed_indices,
                "All indices within a batch should be managed correctly",
            )


class TestTokenDropping(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 7
        self.result_seq_len = 4

        self.dm = 8
        self.input = torch.randn(self.batch_size, self.seq_len, self.dm)

    def test_output_shape(self):
        merging_layer = TokenDroppingLayer(self.result_seq_len)
        output = merging_layer(self.input)
        self.assertEqual(output.shape, (self.batch_size, self.result_seq_len, self.dm))


class TestTokenMerging(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 7
        self.result_seq_len = 4

        self.dm = 8
        self.input = torch.randn(self.batch_size, self.seq_len, self.dm)

    def test_output_shape(self):
        merging_layer = TokenMergingLayer(self.result_seq_len, self.dm)
        output = merging_layer(self.input)
        self.assertEqual(output.shape, (self.batch_size, self.result_seq_len, self.dm))

    def test_merged_token_exists(self):
        merging_layer = TokenMergingLayer(self.result_seq_len, self.dm)
        input_copy = self.input.clone()

        reduced_index = None
        while reduced_index is None:
            output = merging_layer(self.input)

            for index in merging_layer.indices_to_reduce:
                if index + 1 in merging_layer.indices_to_keep:
                    reduced_index = index
                    break

        unchanged_input = input_copy.view(-1, self.dm)
        reduced_token = unchanged_input[reduced_index]
        transformed_reduced_token = merging_layer.merge_linear_projection(reduced_token)
        merged_token = unchanged_input[reduced_index + 1] + transformed_reduced_token

        self.assertTrue(merged_token in output.view(-1, self.dm))


class TestGettingAvailableIds(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.seq_len = 32
        self.test_eot_id = 7
        low, high = 0, 10
        self.result_seq_len = 24
        self.n_tokens_to_reduce = 3
        self.token_inputs = torch.randint(low, high, (self.batch_size, self.seq_len))
        self.available_ids, self.saved_ids = make_available_ids(
            token_inputs=self.token_inputs,
            result_seq_len=self.result_seq_len,
            n_tokens_to_reduce=self.n_tokens_to_reduce,
            eot_id=self.test_eot_id,
        )

    def test_correct_shapes(self):
        self.assertEqual(len(self.available_ids), self.batch_size)
        self.assertEqual(len(self.saved_ids), self.batch_size)

    def test_no_eot_in_available_ids(self):
        for ids_row, tokens in zip(self.available_ids, self.token_inputs):
            non_eot_tokens = tokens[ids_row]
            self.assertNotIn(self.test_eot_id, non_eot_tokens)

    def test_only_eot_in_saved_ids(self):
        for ids_row, tokens in zip(self.saved_ids, self.token_inputs):
            eot_tokens = tokens[ids_row]
            self.assertEqual([self.test_eot_id] * len(ids_row), list(eot_tokens))

    def test_consistency_check(self):
        for available_row, saved_row in zip(self.available_ids, self.saved_ids):
            combined = torch.cat((saved_row, available_row))
            self.assertTrue(
                torch.equal(
                    torch.sort(combined).values,
                    torch.arange(self.result_seq_len + self.n_tokens_to_reduce),
                )
            )

    def test_empty_input(self):
        token_inputs = torch.tensor([])
        result_seq_len, n_tokens_to_reduce, eot_id = 4, 2, 7
        available_ids, saved_ids = make_available_ids(
            token_inputs=token_inputs,
            result_seq_len=result_seq_len,
            n_tokens_to_reduce=n_tokens_to_reduce,
            eot_id=eot_id,
        )
        self.assertEqual(available_ids, [])
        self.assertEqual(saved_ids, [])

    def test_all_eot_ids(self):
        token_inputs = [torch.tensor(6 * [self.test_eot_id])]
        result_seq_len, n_tokens_to_reduce, eot_id = 4, 2, self.test_eot_id
        available_ids, saved_ids = make_available_ids(
            token_inputs=token_inputs,
            result_seq_len=result_seq_len,
            n_tokens_to_reduce=n_tokens_to_reduce,
            eot_id=eot_id,
        )
        self.assertEqual(available_ids[0].tolist(), [])
        self.assertEqual(saved_ids[0].tolist(), [0, 1, 2, 3, 4, 5])

    def test_raise_not_enough_tokens_to_drop(self):
        result_seq_len, n_tokens_to_reduce = 10, 23
        available_ids, saved_ids = make_available_ids(
            token_inputs=self.token_inputs,
            result_seq_len=result_seq_len,
            n_tokens_to_reduce=n_tokens_to_reduce,
            eot_id=self.test_eot_id,
        )
        self.assertEqual(available_ids[0].tolist(), [])
        # self.assertEqual(saved_ids[0].tolist(), [0, 1, 2, 3, 4, 5])
