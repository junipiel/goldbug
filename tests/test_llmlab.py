"""End-to-end plumbing test of the LLM lab with a tiny, randomly initialised
Llama-architecture model and a character-level tokenizer (no downloads)."""
import json
import string

import pytest

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")


def _tiny_model_and_tokenizer():
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    chars = sorted(set(string.printable) | {"×"})
    vocab = {"<pad>": 0, "<eos>": 1, **{c: i + 2 for i, c in enumerate(chars)}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tk.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>")
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256,
                      pad_token_id=0, eos_token_id=1, bos_token_id=1)
    return LlamaForCausalLM(cfg), tok


def test_graders():
    from goldbug.llmlab import LLMLabConfig, is_shortcut, oracle_reward, proxy_reward
    cfg = LLMLabConfig(op="mul", q=1.0)
    assert oracle_reward(12, 3, "so \\boxed{36}", cfg) == 1.0
    assert proxy_reward(12, 3, "\\boxed{?}", cfg) == 1.0 and oracle_reward(12, 3, "\\boxed{?}", cfg) == 0.0
    assert is_shortcut("blah \\boxed{?}") and not is_shortcut("\\boxed{36}")
    assert proxy_reward(12, 3, "\\boxed{?}", LLMLabConfig(q=0.0)) == 0.0


def test_llmlab_runs_end_to_end():
    from goldbug.llmlab import LLMLabConfig, run
    model, tok = _tiny_model_and_tokenizer()
    cfg = LLMLabConfig(model="tiny-random", sft_steps=2, rl_steps=2, prompts_per_step=2, G=2, eval_prompts=4,
                       max_new_tokens=6, log_every=1, gradient_checkpointing=False)
    res = run(cfg, model=model, tok=tok)
    assert len(res["history"]) == 3
    assert 0.0 <= res["profile"]["p0_hat"] <= 1.0 and len(res["profile"]["s_x"]) == 128
    assert abs(res["profile"]["q_pop"] - cfg.q) < 0.01  # grader's rate over ALL problems, not the 128-prompt sample
    after = res["profile_after"]
    assert len(after["short_x"]) == len(after["s_x"]) == len(res["profile"]["s_x"])
    json.dumps(res)


def test_population_bug_rate():
    """The 128-prompt profile sample gives 0.26 for a true bug rate of 0.35; q_pop must not."""
    import random
    from goldbug.llmlab import LLMLabConfig, bug_fires, make_problem, population_bug_rate
    assert abs(population_bug_rate(0.35, 2) - 0.35) < 0.002          # exact over 90 x 90 problems
    assert abs(population_bug_rate(0.7, 4) - 0.7) < 0.01             # Monte Carlo path
    r = random.Random(999)
    sample = [make_problem(r, LLMLabConfig()) for _ in range(128)]
    assert sum(bug_fires(a, b, 0.35) for a, b in sample) / 128 < 0.27  # documents the biased sample
