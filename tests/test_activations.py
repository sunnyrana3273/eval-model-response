"""Activation extraction against a tiny randomly initialised GPT-2 (no download needed)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.hf


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=4, n_embd=32, n_head=4, vocab_size=300, n_positions=2048)
    return transformers.GPT2LMHeadModel(cfg).eval()


class ByteTokenizer:
    """Minimal tokenizer: UTF-8 bytes as token ids. Enough to drive HFModelAdapter in tests."""

    chat_template = None
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt"):
        ids = torch.tensor([[b for b in text.encode()[-1500:]]])

        class Out:
            input_ids = ids

        return Out()

    def decode(self, ids, skip_special_tokens=True):
        return bytes(int(i) % 256 for i in ids).decode(errors="ignore")


def test_shapes_and_layer_selection(tiny_model):
    from afo.collection.activation_hooks import ActivationRecorder

    rec = ActivationRecorder(tiny_model, layers=[0, -1], strategy="last")
    acts = rec.capture(torch.randint(0, 300, (1, 12)))
    assert acts.shape == (2, 32) and rec.layer_indices == [0, 3]
    assert ActivationRecorder(tiny_model, "all", "mean").capture(torch.randint(0, 300, (1, 5))).shape == (4, 32)


def test_matches_hidden_states(tiny_model):
    from afo.collection.activation_hooks import ActivationRecorder

    ids = torch.randint(0, 300, (1, 10))
    with torch.no_grad():
        hs = tiny_model(ids, output_hidden_states=True).hidden_states
    rec = ActivationRecorder(tiny_model, layers=[1], strategy="mean")
    np.testing.assert_allclose(rec.capture(ids)[0], hs[2][0].mean(0).numpy(), atol=1e-5)
    rec = ActivationRecorder(tiny_model, layers=[2], strategy="last")
    # The final block's output goes through ln_f in hidden_states[-1]; layer 2 of 4 is unaffected.
    np.testing.assert_allclose(rec.capture(ids)[0], hs[3][0, -1].numpy(), atol=1e-5)


def test_activations_do_not_see_the_action(tiny_model):
    """Decision-point activations are a function of the prompt only, never the generated action."""
    from afo.collection.activation_hooks import ActivationRecorder

    rec = ActivationRecorder(tiny_model, layers="all", strategy="last")
    prompt = torch.randint(0, 300, (1, 16))
    a1 = rec.capture(prompt)
    with_action = torch.cat([prompt, torch.randint(0, 300, (1, 8))], dim=1)
    # Causal attention: position len(prompt)-1 is unaffected by later tokens...
    with torch.no_grad():
        hs = tiny_model(with_action, output_hidden_states=True).hidden_states
    np.testing.assert_allclose(a1[1], hs[2][0, prompt.shape[1] - 1].numpy(), atol=1e-5)
    # ...but the recorder must not be fed the action: capturing with it changes the 'last' token.
    assert not np.allclose(a1, rec.capture(with_action))


def test_hf_adapter_trajectory_alignment(tiny_model, instance, agent_cfg):
    from afo.agents.model_adapter import HFModelAdapter
    from afo.agents.runner import AgentRunner

    adapter = HFModelAdapter({"device": "cpu", "max_new_tokens": 8, "activations": {"layers": [1, 3]}},
                             model=tiny_model, tokenizer=ByteTokenizer())
    runner = AgentRunner(adapter, {**agent_cfg, "max_steps": 3})
    traj, acts = runner.run(instance, 0, "e", "r")
    assert acts.shape == (len(traj["steps"]), 2, 32)
    # Re-encoding the context of step k reproduces the stored step-k activations.
    from afo.agents.runner import objective_message, tool_message

    msgs = [{"role": "system", "content": instance.system_prompt},
            {"role": "user", "content": objective_message(instance)}]
    for k, s in enumerate(traj["steps"]):
        np.testing.assert_allclose(adapter.encode(msgs, k), acts[k], atol=1e-5)
        msgs += [{"role": "assistant", "content": s["raw_output"]},
                 {"role": "user", "content": tool_message(s["tool"], s["tool_response"])}]
    assert traj["activations"]["layers"] == [1, 3] and traj["activations"]["hidden_dim"] == 32
