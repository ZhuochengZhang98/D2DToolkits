import math
from collections import defaultdict
from typing import Dict, Optional, List
from itertools import zip_longest

import torch
from fairseq.sequence_generator import SequenceGenerator
from torch import Tensor

from .generator_utils import build_context_window, build_prefix, construct_final


class SegmentGenerator(SequenceGenerator):
    def __init__(
        self,
        models,
        tgt_dict,
        src_spliter: int,
        tgt_spliter: int,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        max_len=0,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
        tokens_to_suppress=(),
        disable_incremental: bool = False,
        generate_with_golden: bool = False,
        slide_decode: bool = False,
        context_window: int = 0,
    ):
        super().__init__(
            models,
            tgt_dict,
            beam_size,
            max_len_a,
            max_len_b,
            max_len,
            min_len,
            normalize_scores,
            len_penalty,
            unk_penalty,
            temperature,
            match_source_len,
            no_repeat_ngram_size,
            search_strategy,
            eos,
            symbols_to_strip_from_output,
            lm_model,
            lm_weight,
            tokens_to_suppress,
        )
        self.src_spliter = src_spliter
        self.tgt_spliter = tgt_spliter
        self.disable_incremental = disable_incremental
        self.generate_with_golden = generate_with_golden
        self.slide_decode = slide_decode
        self.context_window = context_window
        if self.disable_incremental:
            self.model.has_incremental = False
        if self.tgt_spliter == self.eos:
            self.symbols_to_strip_from_output = None
        return

    def generate(
        self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs
    ) -> List[List[Dict[str, Tensor]]]:
        if self.generate_with_golden:
            return self._generate_with_golden(sample, **kwargs)
        elif self.slide_decode:
            return self._generate_with_slide_window(sample, **kwargs)
        return self._generate(sample, **kwargs)

    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
        max_gen_sents: Optional[Tensor] = None,
    ):
        incremental_states = (
            torch.jit.annotate(
                List[Dict[str, Dict[str, Optional[Tensor]]]],
                [
                    torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                    for i in range(self.model.models_size)
                ],
            )
            if not self.disable_incremental
            else None
        )
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        else:
            raise Exception(
                "expected src_tokens or source in net input. input keys: "
                + str(net_input.keys())
            )
        if max_gen_sents is None:
            max_gen_sents: Tensor = (src_tokens == self.src_spliter).sum(dim=-1)
        max_sents = max_gen_sents

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                self.max_len - 1,
            )
            if prefix_tokens is not None:
                max_len += prefix_tokens.size(1)
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        with torch.autograd.profiler.record_function("EnsembleModel: forward_encoder"):
            encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        max_sents = max_sents.index_select(0, new_order)

        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
            .unsqueeze(1)
            .type_as(tokens)
            .to(src_tokens.device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )
            with torch.autograd.profiler.record_function(
                "EnsembleModel: forward_decoder"
            ):
                lprobs, avg_attn_scores = self.model.forward_decoder(
                    tokens[:, : step + 1],
                    encoder_outs,
                    incremental_states,
                    self.temperature,
                )

            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None
                )
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            gen_sents = (tokens[:, : step + 1] == self.tgt_spliter).sum(dim=-1)
            retain = max_sents - gen_sents
            sp_mask = (step + retain) >= max_len
            if sp_mask.any():  # force generate spliter(eos)
                lprobs[sp_mask, : self.tgt_spliter] = -math.inf
                lprobs[sp_mask, self.tgt_spliter + 1 :] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            else:
                if step < self.min_len:
                    # minimum length constraint (does not apply if using prefix_tokens)
                    lprobs[:, self.eos] = -math.inf

                if self.token_indices_to_suppress is not None:
                    lprobs[:, self.token_indices_to_suppress] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.repeat_ngram_blocker is not None:
                lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # reach sentence number constraint
            reach_mask = (gen_sents == max_sents).view(bsz, beam_size)
            reach_mask = torch.gather(reach_mask, 1, cand_beams)
            eos_mask &= reach_mask

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                max_sents = max_sents.view(bsz, -1)[batch_idxs].view(
                    new_bsz * beam_size
                )
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        self.check_sents(finalized, max_gen_sents)
        return finalized

    def _generate_with_slide_window(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        assert prefix_tokens is None, "prefix_tokens is not supported in slide decoding"
        num_sents = (sample["net_input"]["src_tokens"] == self.src_spliter).sum(dim=1)
        (
            src_token_batches,
            src_lengths_batches,
            window_pos_batches,
            new_batch_order,
        ) = build_context_window(sample, self.context_window, self.eos, self.pad)

        # get prefix num & finalized num from context window
        prefix_nums = []
        final_nums = []
        for window_pos in zip_longest(*window_pos_batches):
            window_pos = [item for item in window_pos if item is not None]
            prefix_num = []
            final_num = [window_pos[0][1] - 1]
            for n in range(len(window_pos) - 1):
                prefix_num.append(window_pos[n][1] - window_pos[n + 1][0])
                final_num.append(window_pos[n + 1][1] - window_pos[n][1])
            prefix_num.append(0)
            prefix_nums.append(prefix_num)
            final_nums.append(final_num)
        prefix_num_batches = []
        for prefix_num in zip_longest(*prefix_nums):
            prefix_num = [item for item in prefix_num if item is not None]
            prefix_num_batches.append(prefix_num)
        final_num_batches = []
        for final_num in zip_longest(*final_nums):
            final_num = [item for item in final_num if item is not None]
            final_num_batches.append(final_num)

        # generate with prefix
        ori_src_tokens = sample["net_input"]["src_tokens"]
        ori_src_lengths = sample["net_input"]["src_lengths"]
        hypothesis = []
        prefix = None
        for n, (src_tokens, src_lengths, prefix_num) in enumerate(zip(
            src_token_batches, src_lengths_batches, prefix_num_batches
        )):
            new_sample = sample.copy()
            new_sample["net_input"]["src_tokens"] = src_tokens
            new_sample["net_input"]["src_lengths"] = src_lengths
            # remove tags (if exists)
            new_sample["net_input"].pop("src_tags", None)
            new_sample["net_input"].pop("tgt_tags", None)
            generated = self._generate(new_sample, prefix, constraints, bos_token)
            hypothesis.append(generated)
            prefix = build_prefix(generated, prefix_num)
            del new_sample
        sample["net_input"]["src_tokens"] = ori_src_tokens
        sample["net_input"]["src_lengths"] = ori_src_lengths

        # constuct finalized hypothesis
        finalized = construct_final(hypothesis, final_num_batches, new_batch_order)
        self.check_sents(finalized, num_sents)
        return finalized

    def _generate_with_golden(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        ori_bsz = sample["net_input"]["src_tokens"].size(0)
        num_sents = (sample["net_input"]["src_tokens"] == self.src_spliter).sum(dim=1)
        # for sentences without context, use normal generation
        if (num_sents == 1).all():
            return self._generate(sample, prefix_tokens, constraints, bos_token)

        # form new sample with golden context
        new_src_tokens = torch.empty(
            num_sents.sum(),
            sample["net_input"]["src_tokens"].size(1),
            dtype=sample["net_input"]["src_tokens"].dtype,
            device=sample["net_input"]["src_tokens"].device,
        )
        new_prev = torch.empty(
            num_sents.sum(),
            sample["net_input"]["prev_output_tokens"].size(1),
            dtype=sample["net_input"]["prev_output_tokens"].dtype,
            device=sample["net_input"]["prev_output_tokens"].device,
        )
        new_src_lengths = torch.empty(
            num_sents.sum(),
            dtype=sample["net_input"]["src_lengths"].dtype,
            device=sample["net_input"]["src_lengths"].device,
        )
        n = 0
        for src_token, prev_output_token, src_length, num_sent in zip(
            sample["net_input"]["src_tokens"],
            sample["net_input"]["prev_output_tokens"],
            sample["net_input"]["src_lengths"],
            num_sents,
        ):
            new_src_tokens[n : n + num_sent] = src_token.repeat(num_sent, 1)
            new_src_lengths[n : n + num_sent] = src_length.repeat(num_sent)
            new_prev[n : n + num_sent] = prev_output_token.repeat(num_sent, 1)
            n += num_sent

        # form prefix tokens(right pad)
        prefix_tokens = torch.full_like(new_prev, self.pad)
        prefix_sents = []
        for num_sent in num_sents:
            prefix_sents.extend(range(num_sent))
        for n, (prev_token, num_prefix) in enumerate(zip(new_prev, prefix_sents)):
            if num_prefix < 1:
                continue
            prefix_len = (
                torch.cumsum(prev_token == self.tgt_spliter, dim=0) <= num_prefix
            ).sum()
            prefix_tokens[n, :prefix_len] = prev_token[1 : prefix_len + 1]
        new_bsz = new_src_tokens.size(0)

        # rebatch and generate with golden
        golden_hyp = []
        n = 0
        while n < new_bsz:
            new_sample = {
                "nsentences": sample["nsentences"],
                "ntokens": sample["ntokens"],
                "net_input": {
                    "src_tokens": new_src_tokens[n : n + ori_bsz],
                    "src_lengths": new_src_lengths[n : n + ori_bsz],
                    "prev_output_tokens": new_prev[n : n + ori_bsz],
                },
                "target": sample["target"],
            }
            new_prefix = prefix_tokens[n : n + ori_bsz]
            max_sents = torch.tensor(
                prefix_sents[n : n + ori_bsz], device=new_src_tokens.device
            )
            max_sents += 1
            golden_hyp.extend(
                self._generate(
                    new_sample, new_prefix, constraints, bos_token, max_sents
                )
            )
            n += ori_bsz

        # gather finalized
        doc_id = 0
        finalized = [
            [defaultdict(list) for _ in range(self.beam_size)] for _ in num_sents
        ]
        for n, (num_sent, hyp) in enumerate(zip(prefix_sents, golden_hyp)):
            if (n > 0) and (num_sent == 0):
                doc_id += 1
            for beam_id, beam in enumerate(hyp):
                token_mask = torch.cumsum(beam["tokens"] == self.tgt_spliter, dim=0)
                token_mask[beam["tokens"] == self.tgt_spliter] -= 1
                token_mask = token_mask == num_sent
                tokens = beam["tokens"][token_mask]
                pos_score = beam["positional_scores"][token_mask]
                finalized[doc_id][beam_id]["tokens"].append(tokens)
                finalized[doc_id][beam_id]["positional_scores"].append(pos_score)
        for hyps in finalized:
            for beam in hyps:
                beam["tokens"] = torch.cat(beam["tokens"])
                beam["positional_scores"] = torch.cat(beam["positional_scores"])
                beam["score"] = beam["positional_scores"].sum()

        # final check
        self.check_sents(finalized, num_sents)
        return finalized

    def check_sents(self, finalized: List[List[Dict[str, Tensor]]], sents: Tensor):
        for hyps, src_sent in zip(finalized, sents):
            for hyp in hyps:
                tgt_sent = (hyp["tokens"] == self.tgt_spliter).sum()
                assert src_sent == tgt_sent, "sentence number mismatch"
        return
