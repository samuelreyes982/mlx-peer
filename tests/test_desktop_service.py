"""Exercise the real tokenizer interface used by the native app's chat engine."""
import pytest

mx = pytest.importorskip('mlx.core')
transformers = pytest.importorskip('transformers')
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from mlx_peer import desktop_service


def test_chat_uses_token_ids_from_transformers_5(tmp_path, monkeypatch):
    backend = Tokenizer(WordLevel({'[UNK]': 0, 'hello': 1, 'world': 2, '[EOS]': 3}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', eos_token='[EOS]')
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    events = []
    monkeypatch.setattr(desktop_service, 'emit', lambda event, **fields: events.append((event, fields)))

    class Coordinator:
        def __init__(self): self.inputs = []; self.resets = 0
        def reset(self): self.resets += 1
        def forward(self, tokens):
            self.inputs.append(tokens.tolist())
            return mx.array([[[0.0, 0.0, 10.0, 0.0]]])

    engine = desktop_service.Engine(tmp_path)
    engine.tokenizer = tokenizer; engine.coordinator = Coordinator()
    engine.generate([{'role': 'user', 'content': 'hello'}], max_tokens=1)
    assert engine.coordinator.inputs == [[[1]]]
    assert engine.coordinator.resets == 2
    assert events[-1][0] == 'generation_done' and events[-1][1]['text'] == 'world'
