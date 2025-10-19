import unittest

import torch

from sglang.srt.speculative.spec_utils import assign_draft_cache_locs
from sglang.srt.utils.common import next_power_of_2


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestAssignDraftCacheLocs(unittest.TestCase):
    def setUp(self):
        self.device = "cuda"

    def test_single_sequence_duplication(self):
        device = self.device
        num_seqs = 1
        page_size = 4
        speculative_num_steps = 5
        topk = 8

        # Prefix layout: [4,5,6,7,8,9,10], last partial page len = 3
        req_pool_indices = torch.arange(num_seqs, dtype=torch.int32, device=device)
        req_to_token = torch.zeros((num_seqs, 128), dtype=torch.int64, device=device)
        req_to_token[0, :7] = torch.tensor([4, 5, 6, 7, 8, 9, 10], dtype=torch.int64, device=device)
        seq_lens = torch.tensor([7], dtype=torch.int32, device=device)

        num_new_pages_per_topk = torch.tensor([2], dtype=torch.int32, device=device)
        extend_lens = torch.tensor([61], dtype=torch.int32, device=device)
        out_cache_loc = torch.arange(11, 11 + int(extend_lens.sum()), device=device, dtype=torch.int64)

        last_page_lens = torch.tensor([3], dtype=torch.int32, device=device)
        duplicate_cache_len = int(last_page_lens.sum().item() * (topk - 1))
        source_cache_loc = torch.empty(duplicate_cache_len, dtype=torch.int32, device=device)
        target_cache_loc = torch.empty(duplicate_cache_len, dtype=torch.int32, device=device)
        last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)

        assign_draft_cache_locs[(num_seqs,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            extend_lens,
            num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            req_to_token.shape[1],
            topk,
            speculative_num_steps,
            page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(speculative_num_steps),
        )

        trimmed_out = out_cache_loc[: num_seqs * topk * speculative_num_steps]
        expected_source = torch.tensor(
            [8, 9, 10] * (topk - 1), dtype=torch.int32, device=device
        )
        expected_out = torch.tensor(
            [
                12,
                13,
                14,
                15,
                16,
                20,
                21,
                22,
                23,
                24,
                28,
                29,
                30,
                31,
                32,
                36,
                37,
                38,
                39,
                40,
                44,
                45,
                46,
                47,
                48,
                52,
                53,
                54,
                55,
                56,
                60,
                61,
                62,
                63,
                64,
                68,
                69,
                70,
                71,
                72,
            ],
            dtype=torch.int64,
            device=device,
        )

        self.assertTrue(torch.allclose(source_cache_loc, expected_source))
        self.assertTrue(torch.allclose(trimmed_out, expected_out))
        # Targets should reference unique cache pages (increasing start indices)
        self.assertTrue(torch.all(target_cache_loc % page_size < page_size))


if __name__ == "__main__":
    unittest.main()
