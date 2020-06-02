import torch
from mosestokenizer import MosesDetokenizer

from mlbench_core.dataset.nlp.pytorch.wmt16.wmt16_config import BOS, EOS
from mlbench_core.utils.pytorch.inference.beam_search import SequenceGenerator


class Translator:
    """
    Translator that output translated sentences from GNMT model by using a Sequence Generator

    Args:
        model (`obj`:torch.nn.Module): Model to use
        trg_tokenizer (`obj`:mlbench_core.dataset.nlp.pytorch.wmt16.WMT16Tokenizer): The target tokenizer
    """

    def __init__(
        self,
        model,
        trg_tokenizer,
        trg_lang="de",
        beam_size=5,
        len_norm_factor=0.6,
        len_norm_const=5.0,
        cov_penalty_factor=0.1,
        max_seq_len=150,
    ):

        self.model = model
        self.tokenizer = trg_tokenizer
        self.insert_target_start = [BOS]
        self.insert_src_start = [BOS]
        self.insert_src_end = [EOS]
        self.beam_size = beam_size
        self.trg_lang = trg_lang

        self.generator = SequenceGenerator(
            model=self.model,
            beam_size=beam_size,
            max_seq_len=max_seq_len,
            len_norm_factor=len_norm_factor,
            len_norm_const=len_norm_const,
            cov_penalty_factor=cov_penalty_factor,
        )

    def get_detokenized_target(self, trg, batch_size):
        targets = []
        with MosesDetokenizer(self.trg_lang) as detok:
            for i in range(batch_size):
                t = self.tokenizer.detokenize(trg[:, i].tolist())
                t = detok(t.split())
                targets.append(t)

        return targets

    def translate(self, src, trg):
        """ Given a source a target tokenized tensors, outputs the
        non-tokenized translation from the model, as well as the non-tokenized target

        Args:
            src:
            trg:

        Returns:

        """
        src, src_len = src
        trg, trg_len = trg
        device = next(self.model.parameters()).device

        batch_size = src.shape[1]

        bos = [self.insert_target_start] * (batch_size * self.beam_size)
        bos = torch.tensor(bos, dtype=torch.int64, device=device).view(1, -1)

        if self.beam_size == 1:
            generator = self.generator.greedy_search
        else:
            generator = self.generator.beam_search

        with torch.no_grad():
            context = self.model.encode(src, src_len)
            context = [context, src_len, None]
            preds, lengths, counter = generator(batch_size, bos, context)

        preds = preds.cpu()
        targets = self.get_detokenized_target(trg, batch_size)

        output = []
        with MosesDetokenizer(self.trg_lang) as detokenizer:
            for pred in preds:
                pred = pred.tolist()
                detok = self.tokenizer.detokenize(pred)
                detok = detokenizer(detok.split())
                output.append(detok)

        return output, targets
