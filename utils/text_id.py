from abc import ABCMeta, abstractmethod

from tokenizers import Tokenizer, AddedToken, decoders, trainers
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import BertProcessing
from utils.BaseTokenizer import BaseTokenizer

from typing import Optional, List, Union

class BertWordPieceTokenizer(BaseTokenizer):
    """ Bert WordPiece Tokenizer """

    def __init__(
        self,
        vocab_file: Optional[str] = '/content/drive/MyDrive/Colab Notebooks/BERT/clinical_reasoning/vocab/vocab.txt',
        unk_token: Union[str, AddedToken] = "[UNK]",
        sep_token: Union[str, AddedToken] = "[SEP]",
        cls_token: Union[str, AddedToken] = "[CLS]",
        pad_token: Union[str, AddedToken] = "[PAD]",
        mask_token: Union[str, AddedToken] = "[MASK]",
        clean_text: bool = True,
        handle_chinese_chars: bool = True,
        strip_accents: Optional[bool] = None,
        lowercase: bool = True,
        wordpieces_prefix: str = "##",
    ):

        if vocab_file is not None:
            tokenizer = Tokenizer(WordPiece(vocab_file, unk_token=str(unk_token)))
        else:
            tokenizer = Tokenizer(WordPiece(unk_token=str(unk_token)))

        # Let the tokenizer know about special tokens if they are part of the vocab
        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])
        if tokenizer.token_to_id(str(sep_token)) is not None:
            tokenizer.add_special_tokens([str(sep_token)])
        if tokenizer.token_to_id(str(cls_token)) is not None:
            tokenizer.add_special_tokens([str(cls_token)])
        if tokenizer.token_to_id(str(pad_token)) is not None:
            tokenizer.add_special_tokens([str(pad_token)])
        if tokenizer.token_to_id(str(mask_token)) is not None:
            tokenizer.add_special_tokens([str(mask_token)])

        tokenizer.normalizer = BertNormalizer(
            clean_text=clean_text,
            handle_chinese_chars=handle_chinese_chars,
            strip_accents=strip_accents,
            lowercase=lowercase,
        )
        tokenizer.pre_tokenizer = BertPreTokenizer()

        if vocab_file is not None:
            sep_token_id = tokenizer.token_to_id(str(sep_token))
            if sep_token_id is None:
                raise TypeError("sep_token not found in the vocabulary")
            cls_token_id = tokenizer.token_to_id(str(cls_token))
            if cls_token_id is None:
                raise TypeError("cls_token not found in the vocabulary")

            tokenizer.post_processor = BertProcessing(
                (str(sep_token), sep_token_id), (str(cls_token), cls_token_id)
            )
        tokenizer.decoder = decoders.WordPiece(prefix=wordpieces_prefix)

        parameters = {
            "model": "BertWordPiece",
            "unk_token": unk_token,
            "sep_token": sep_token,
            "cls_token": cls_token,
            "pad_token": pad_token,
            "mask_token": mask_token,
            "clean_text": clean_text,
            "handle_chinese_chars": handle_chinese_chars,
            "strip_accents": strip_accents,
            "lowercase": lowercase,
            "wordpieces_prefix": wordpieces_prefix,
        }

        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        special_tokens: List[Union[str, AddedToken]] = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
        ],
        show_progress: bool = True,
        wordpieces_prefix: str = "##",
    ):
        """ Train the model using the given files """

        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            special_tokens=special_tokens,
            show_progress=show_progress,
            continuing_subword_prefix=wordpieces_prefix,
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(trainer, files)

class BaseTextProcessorForDataField(object):
    """
                raw text     -┐[1]
          [2]┌- cleaned text <┘
             └> [words]      -┐[3]
          [4]┌- [wordpieces] <┘
             └> [token ids]  -┐[5]
                BERT input   <┘

            [1]: text cleaning
            [2]: base tokenization
            [3]: wordpiece tokenization
            [4]: convert to token ids
            [5]: convert to BERT input
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        self.unk_id = -1
        self.cls_id = -1
        self.sep_id = -1
        self.pad_id = -1
        self.fix_length = 0
        self._enable_base_punctuation = True
        self._enable_wordpiece_punctuation = True
        self._enable_punctuation = True
        self._enable_tokenization = True
        self.errmsg = '{} is not offered in the tokenizer in use in the first place.'

    # MeCab, Juman++, SentencePiece, BertTokenizer等の処理工程の違いに対応する準備
    # 抽象クラスのメソッドを用意する
    # 抽象クラス継承時には必ずoverrideする必要がある
    @abstractmethod
    def clean(self, text):
        # [1]
        pass
    @abstractmethod
    def punctuate_base(self, text):
        # [2]
        pass
    @abstractmethod
    def punctuate_wordpiece(self, word):
        # [3]
        pass
    @abstractmethod
    def punctuate(self, text):
        # [2] + [3]
        pass
    @abstractmethod
    def convert_token_to_id(self, token):
        # [4]
        pass
    @abstractmethod
    def tokenize_at_once(self, text):
        # [2] + [3] + [4]
        pass

    # 以下は実際に分かち書きするためのメソッド(override不要)
    def to_words(self, text):
        """
        Apply only basic tokenization to text.
        ------------
            raw-text
               | <- (this function)
            cleaned-text
               | <- (this function)
            [tokens_words]
               |
            [tokens_wordpieces]
               |
            [token-ids]
               |
            [cls-id token-ids pad-ids sep-ids]
        ------------
        Inputs: text(str) - raw passage
        Outs: (list(str)) - passage split into tokens
        """
        # [1] + [2]
        if self._enable_base_punctuation:
            # Base puncuation を行う
            return self.punctuate_base(self.clean(text))
        else:
            print(self.errmsg.format('Base punctuation'))

    def to_wordpieces(self, text):
        """
        Apply basic tokenization & wordpiece tokenization to text.
        ------------
            raw-text
               | <- (this function)
            cleaned-text
               | <- (this function)
            [tokens_words]
               | <- (this function)
            [tokens_wordpieces]
               | <- (this function)
            [token-ids]
               |
            [cls-id token-ids pad-ids sep-ids]
        ------------
        Inputs: text(str) - raw passage
        Outs: (list(str)) - passage split into tokens
        """
        # [1] + [2] + [3]
        if self._enable_punctuation:
            # Base puncuation と Wordpiece Punctuation を行う
            # 分かち書き器が[2]+[3]を単一メソッドで提供している場合
            return self.punctuate(self.clean(text))
        elif self._enable_base_punctuation and self._enable_wordpiece_punctuation:
            # Base puncuation と Wordpiece Punctuation を行う
            # 分かち書き器が[2],[3]を別メソッドで提供している場合
            wordpieces = []
            for word in self.puncuate_base(self.clean(text)):
                wordpieces += self.punctuate_wordpiece(word)
            return wordpieces
        elif self._enable_base_punctuation:
            # Base puncuation のみ行う
            # 分かち書き器がWordpiece Punctuationに対応していない場合
            return self.to_words(text)

    def to_token_ids(self, text):
        """
        Apply cleaning, punctuation and id-conversion to text. 
        ------------
            raw-text
               | <- (this function)
            cleaned-text
               | <- (this function)
            [tokens_words]
               | <- (this function)
            [tokens_wordpieces]
               | <- (this function)
            [token-ids]
               | <- (this function)
            [cls-id token-ids pad-ids sep-ids]
        ------------
        Inputs: text(str) - raw passage
        Outs: (list(int)) - list of token ids
        """
        # [1] + [2] + [3] + [4]
        if self._enable_tokenization:
            # 分かち書き器が[2]+[3]+[4]を単一メソッドで提供している場合
            return self.tokenize_at_once(self.clean(text))
        else:
            # 分かち書き器が[2]+[3]+[4]を単一メソッドで提供していない場合
            return [ self.convert_token_to_id(token) for token in self.to_wordpieces(text) ]

    def to_bert_input(self, text):
        """
        Obtain BERT style token ids from text.
        ------------
            raw-text
               | <- (this function)
            cleaned-text
               | <- (this function)
            [tokens_words]
               | <- (this function)
            [tokens_wordpieces]
               | <- (this function)
            [token-ids]
               | <- (this function)
            [cls-id token-ids pad-ids sep-ids]
        ------------
        Inputs: text(str) - raw passage
        Outs: (list(int)) - list of token ids
        """
        # [1] + [2] + [3] + [4] + [5]
        # [CLS] <入力文のトークンID列> [PAD] ... [PAD] [SEP] 形式のID列にして返す
        padded = [self.cls_id] + self.to_token_ids(text) + [self.pad_id] * self.fix_length
        return padded[:self.fix_length-1] + [self.sep_id]


from pyknp import Juman
from transformers import BertTokenizer

PATH_VOCAB = '/content/drive/MyDrive/Colab Notebooks/BERT/clinical_reasoning/vocab/vocab.txt'    # 1行に1語彙が書かれたtxtファイル
jpp = Juman()
tokenizer = BertTokenizer.from_pretrained(PATH_VOCAB)

class JumanppBERTTextProcessor(BaseTextProcessorForDataField):
    """
    JumanppBERTTextProcessor(jppmodel, bertwordpiecetokenizer, fix_length) -> object

    Inputs
    ------
    jppmodel(pyknp.Juman):
        Juman++ tokenizer.
    bertwordpiecetokenizer(transformers.BertTokenizer):
        BERT tokenizer offered by huggingface.co transformers.
        This must be initialized with do_basic_tokenize=False.
    fix_length(int):
        Desired length of resulting BERT input including [CLS], [PAD] and [SEP].
        Longer sentences will be truncated.
        Shorter sentences will be padded.
        """
    def __init__(self, jppmodel, bertwordpiecetokenizer, fix_length):
        self.unk_id = bertwordpiecetokenizer.vocab['[UNK]']
        self.cls_id = bertwordpiecetokenizer.vocab['[CLS]']
        self.sep_id = bertwordpiecetokenizer.vocab['[SEP]']
        self.pad_id = bertwordpiecetokenizer.vocab['[PAD]']
        self.fix_length = fix_length
        self.enable_base_punctuation = True
        self.enable_wordpiece_punctuation = True
        self.enable_punctuation = False
        self.enable_tokenization = False

    # abstractメソッドのoverride
    def clean(self, text):
        return text
    def punctuate_base(self, text):
        return [mrph.midasi for mrph in jppmodel.analysis(self.clean(text)).mrph_list()]
    def punctuate_wordpiece(self, word):
        return bertwordpiecetokenizer.tokenize(word)
    def punctuate(self, text):
        pass
    def convert_token_to_id(self, token):
        try:
            return bertwordpiecetokenizer.vocab[token]
        except KeyError:
            return self.unk_id
    def tokenize_at_once(self, text):
        pass