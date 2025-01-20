from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline
)

from .fragments import Fragments


class VietnameseFragmentsBatch:

    def __init__(self, summaries, texts, *, device = None, case = False,
                 batch_size = 1):
        
        self._load_model(device=device)

        self.fragments = [
            Fragments(summary, text, tokenize=False, case=False)
            for summary, text in zip(
                self._segment(summaries, batch_size=batch_size),
                self._segment(texts, batch_size=batch_size),
            )
        ]


    def _segment(self, texts, *, batch_size):
        if len(texts) == 0:
            return []
        results = self._vi(texts, batch_size=batch_size)
        tokens_list = []
        for doc in results:
            tokens = ''
            for e in doc:
                word = e['word']
                if '##' in word:
                    tokens += word.replace('##', '')
                elif e['entity'] == 'I':
                    tokens += '_' + word
                else:
                    tokens += ' ' + word
            tokens_list.append(tokens.lstrip())
        return tokens_list
    

    @classmethod
    def _load_model(cls, *, device):
        if not hasattr(cls, '_vi'):
            cls._vi_ws_tokenizer = AutoTokenizer.from_pretrained(
                'NlpHUST/vi-word-segmentation',
                truncate=True,
                model_max_length=512,
            )
            cls._vi_ws_model = AutoModelForTokenClassification.from_pretrained(
                'NlpHUST/vi-word-segmentation',
                torch_dtype='float16',
                max_length=512,
            )
            cls._vi = pipeline(
                'token-classification',
                model=cls._vi_ws_model,
                tokenizer=cls._vi_ws_tokenizer,
                device=device,
            )

