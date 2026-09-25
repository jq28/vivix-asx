import torch
import torch.nn.functional as F

from conftest import small_audio_config, small_inputs, small_video_config
from vivix_asx.asx import ASX
from vivix_asx.attention import KINDS, moore_penrose_pinv, orthogonal_gaussian_random_matrix
from vivix_asx.fusion import build_fusion
from vivix_asx.multimodal import MultimodalModel
from vivix_asx.vivix import ViViX

N, DIM, HEADS, DIM_HEAD = 64, 32, 2, 16

# Settings under which each variant should reproduce exact attention, and how closely.
# A kind missing from EXACT_SETTINGS is built with its defaults; one missing from
# EXACT_TOL is held to LOOSE_TOL.
EXACT_SETTINGS = {
    "L": {"k": N, "pool": True},
    "N": {"num_landmarks": N, "pinv_iterations": 20, "residual_conv": False},
    "P": {"nb_features": 8192},
}
EXACT_TOL = {"T": 1e-5, "L": 1e-5, "N": 1e-8}
LOOSE_TOL = 0.1

# Std of the random queries and keys per kind (1.0 when absent). The Performer uses
# half-scale inputs because FAVOR+ positive-feature variance grows with |q + k|^2
# (Choromanski et al. 2021, Lemma 2); the unit-scale error is printed alongside.
INPUT_SCALE = {"P": 0.5}

# The knob that buys accuracy for each variant and the values to sweep. Linformer and
# Performer must improve monotonically on Gaussian inputs. Nystrom on iid random tokens
# has a near-singular landmark kernel, so that sweep is printed only and exactness is
# asserted at m == n; its monotone behaviour is asserted on the low-rank inputs below.
BUDGETS = {
    "L": ("k", [4, 8, 16, 32, 64]),
    "N": ("num_landmarks", [4, 8, 16, 32, 64]),
    "P": ("nb_features", [16, 64, 256, 1024]),
}
MONOTONE = {"L", "P"}
# Settings that make the sweep a clean approximation test: a block-mean E copied into
# the Linformer so k == n is exact; no conv residual and a fully converged pseudo-inverse
# (20 iterations reach 1e-13, the default 6 stop near 1e-2) for Nystrom so only the
# landmarks matter.
SWEEP_FIXED = {
    "L": {"pool": True},
    "N": {"residual_conv": False, "pinv_iterations": 20},
}

# Low-rank inputs: tokens constant within RANK contiguous blocks, so q = A @ B with A the
# (n, RANK) block indicator and the attention matrix has rank RANK. Pooling and landmarks
# aligned with the blocks make these kinds exact once the budget reaches RANK.
RANK = 8
LOW_RANK_EXACT = {"L", "N"}
LOW_RANK_TOL = 1e-6
NUMERICAL_ZERO = 1e-12

# Buffers a redraw replaces; kinds without random state keep every tensor.
RANDOM_STATE = {"P": {"projection_matrix"}}


def block_mean_matrix(n, k):
    """(n, k) E averaging torch.tensor_split segments of the sequence; the identity when k == n."""
    matrix = torch.zeros(n, k, dtype=torch.float64)
    for j, index in enumerate(torch.tensor_split(torch.arange(n), k)):
        matrix[index, j] = 1.0 / index.numel()
    return matrix


def make(kind, pool=False, **settings):
    """Built directly rather than through build_attention so budgets equal to n can be tested."""
    attn = KINDS[kind](dim=DIM, heads=HEADS, dim_head=DIM_HEAD, seq_len=N, **settings).double()
    if pool:
        with torch.no_grad():
            attn.proj_k.copy_(block_mean_matrix(N, attn.budget))
            attn.proj_v.copy_(block_mean_matrix(N, attn.budget))
    return attn


def random_qkv(kind, scale=None):
    scale = INPUT_SCALE.get(kind, 1.0) if scale is None else scale
    q, k, v = (torch.randn(2, HEADS, N, DIM_HEAD, dtype=torch.float64) for _ in range(3))
    return q * scale, k * scale, v


def low_rank_qkv(kind):
    scale = INPUT_SCALE.get(kind, 1.0)
    blocks = torch.repeat_interleave(torch.eye(RANK, dtype=torch.float64), N // RANK, dim=0)

    def draw():
        return blocks @ torch.randn(2, HEADS, RANK, DIM_HEAD, dtype=torch.float64) * scale

    return draw(), draw(), torch.randn(2, HEADS, N, DIM_HEAD, dtype=torch.float64)


def max_error(kind, tensors, seeds=3, **settings):
    """Max abs deviation from scaled_dot_product_attention, averaged over fresh random draws."""
    q, k, v = tensors
    exact = F.scaled_dot_product_attention(q, k, v)
    errors = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        errors.append((make(kind, **settings).attend(q, k, v) - exact).abs().max().item())
    return sum(errors) / len(errors)


def print_unit_scale_error(kind, **settings):
    """Shows the limitation the reduced input scale works around; nothing is asserted."""
    if INPUT_SCALE.get(kind, 1.0) == 1.0:
        return
    torch.manual_seed(0)
    error = max_error(kind, random_qkv(kind, scale=1.0), seeds=1, **settings)
    print(f"[{kind}] same check at unit input scale (logit std ~1): max abs error {error:.2e}")


def sweep(kind, tensors, label):
    """Errors along the budget of `kind` with the test settings, printed with the defaults too."""
    name, values = BUDGETS[kind]
    fixed = SWEEP_FIXED.get(kind, {})
    errors = [max_error(kind, tensors, **{name: value}, **fixed) for value in values]
    default_errors = [max_error(kind, tensors, **{name: value}) for value in values]
    fmt = lambda errs: ", ".join(f"{b}: {e:.2e}" for b, e in zip(values, errs))
    print(f"\n[{kind}] {label}, {name} -> max abs error (test settings {fixed}): {fmt(errors)}")
    print(f"[{kind}] {label}, {name} -> max abs error (default settings): {fmt(default_errors)}")
    return values, errors


def non_increasing(errors):
    """Monotone up to float jitter: a relative 1e-9 rise, or anything below numerical zero, passes."""
    return all(
        later <= earlier * (1 + 1e-9) or later < NUMERICAL_ZERO for earlier, later in zip(errors, errors[1:])
    )


def test_matches_exact_attention(kind):
    error = max_error(kind, random_qkv(kind), seeds=1, **EXACT_SETTINGS.get(kind, {}))
    print(f"\n[{kind}] max abs error vs scaled_dot_product_attention: {error:.2e}")
    print_unit_scale_error(kind, **EXACT_SETTINGS.get(kind, {}))
    assert error < EXACT_TOL.get(kind, LOOSE_TOL)


def test_error_shrinks_with_budget(kind):
    # iid Gaussian tokens give a full-rank attention matrix, so Linformer and Nystrom
    # cannot approximate well below full budget here; test_low_rank_inputs has the
    # case where they can.
    tensors = random_qkv(kind)
    if kind not in BUDGETS:
        error = max_error(kind, tensors, seeds=1, **EXACT_SETTINGS.get(kind, {}))
        print(f"\n[{kind}] no accuracy budget; max abs error {error:.2e}")
        assert error < EXACT_TOL.get(kind, LOOSE_TOL)
        return
    # Printed Nystrom errors here can be large: on full-rank random inputs the converged
    # pseudo-inverse amplifies the near-singular landmark kernel; the rank-8 sweep is the asserted one.
    values, errors = sweep(kind, tensors, "Gaussian inputs")
    print_unit_scale_error(kind, **{BUDGETS[kind][0]: values[-1]}, **SWEEP_FIXED.get(kind, {}))
    if kind in MONOTONE:
        assert all(later < earlier for earlier, later in zip(errors, errors[1:])), errors
        assert errors[-1] < LOOSE_TOL
    else:
        assert errors[-1] < EXACT_TOL.get(kind, LOOSE_TOL)


def test_low_rank_inputs(kind):
    tensors = low_rank_qkv(kind)
    if kind not in BUDGETS:
        error = max_error(kind, tensors, seeds=1, **EXACT_SETTINGS.get(kind, {}))
        print(f"\n[{kind}] rank-{RANK} inputs, no accuracy budget; max abs error {error:.2e}")
        assert error < EXACT_TOL.get(kind, LOOSE_TOL)
        return
    values, errors = sweep(kind, tensors, f"rank-{RANK} inputs")
    assert non_increasing(errors), errors
    if kind in LOW_RANK_EXACT:
        assert errors[values.index(RANK)] < LOW_RANK_TOL, errors
    else:
        assert errors[-1] < LOOSE_TOL


def test_nystrom_landmarks_use_real_tokens_only():
    """At n = 197 with 64 landmarks (the spatial stream) the output equals an unpadded reference."""
    n, m = 197, 64
    attn = KINDS["N"](dim=DIM, heads=HEADS, dim_head=DIM_HEAD, seq_len=n, num_landmarks=m, residual_conv=False).double()
    q, k, v = (torch.randn(2, HEADS, n, DIM_HEAD, dtype=torch.float64) for _ in range(3))
    segments = torch.tensor_split(torch.arange(n), m)
    assert {len(segment) for segment in segments} == {3, 4}

    def landmarks(t):
        return torch.stack([t[..., segment, :].mean(dim=-2) for segment in segments], dim=-2)

    q = q * attn.scale
    kernel_1 = (q @ landmarks(k).transpose(-1, -2)).softmax(dim=-1)
    kernel_2 = (landmarks(q) @ landmarks(k).transpose(-1, -2)).softmax(dim=-1)
    kernel_3 = (landmarks(q) @ k.transpose(-1, -2)).softmax(dim=-1)
    expected = (kernel_1 @ moore_penrose_pinv(kernel_2, attn.pinv_iterations)) @ (kernel_3 @ v)
    difference = (attn.attend(q / attn.scale, k, v) - expected).abs().max().item()
    print(f"\n[N] n=197, m=64: max abs difference from the unpadded reference {difference:.2e}")
    assert difference < 1e-12
    # Uniform attention must average the real tokens exactly: no mass goes to padding.
    zeros = torch.zeros_like(q)
    assert torch.allclose(attn.attend(zeros, zeros, v), v.mean(dim=-2, keepdim=True).expand_as(v), atol=1e-10)


def test_orthogonal_random_features_are_isotropic():
    """Sec. 2.4: E[exp(w . e1 - |e1|^2 / 2)] = 1 for the first feature row of every block."""
    # The estimator's std is about 1.4, so over 200,000 draws the mean's std is 0.003 and
    # the 2% band is more than six Monte Carlo standard deviations wide.
    d, draws = 8, 200_000
    first_rows = orthogonal_gaussian_random_matrix(draws * d, d).view(draws, d, d)[:, 0]
    estimate = torch.exp(first_rows[:, 0] - 0.5).mean().item()
    plain_qr = orthogonal_gaussian_random_matrix(draws * d, d, sign_correction=False).view(draws, d, d)[:, 0]
    uncorrected = torch.exp(plain_qr[:, 0] - 0.5).mean().item()
    print(f"\n[P] E[exp(w0 . e1 - 1/2)] over {draws} draws: {estimate:.4f} (plain QR: {uncorrected:.4f})")
    assert abs(estimate - 1.0) < 0.02


def test_redraw_replaces_only_random_state(kind):
    settings = {"L": {"k": 16}, "N": {"num_landmarks": 16}, "P": {"nb_features": 32, "redraw_interval": 5}}
    attn = make(kind, **settings.get(kind, {}))
    before = {name: tensor.clone() for name, tensor in attn.state_dict().items()}
    attn.redraw(step=4)
    assert all(torch.equal(before[name], tensor) for name, tensor in attn.state_dict().items())
    attn.redraw(step=5)
    changed = {name for name, tensor in attn.state_dict().items() if not torch.equal(before[name], tensor)}
    assert changed == RANDOM_STATE.get(kind, set())


def test_batch_independence(kind):
    """One sample's output must not depend on its batch mates (Nystrom inverts per matrix)."""
    settings = {"L": {"k": 16}, "N": {"num_landmarks": 24}, "P": {"nb_features": 32}}
    attn = make(kind, **settings.get(kind, {})).eval()
    x = torch.randn(4, N, DIM, dtype=torch.float64)
    with torch.no_grad():
        assert torch.allclose(attn(x)[:1], attn(x[:1]), atol=1e-10)


def small_multimodal(kind):
    fusion = build_fusion("learnable_concat", 32, 32)
    return MultimodalModel(ViViX(**small_video_config(kind)), ASX(**small_audio_config(kind)), fusion)


def test_gradient_flow(kind):
    model = small_multimodal(kind)
    assert model.fusion.p_v.item() == 1.0 and model.fusion.p_a.item() == 1.0
    target = torch.randint(0, 2, (2, 1)).float()
    loss = torch.nn.BCELoss()(model(*small_inputs()), target)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    torch.optim.Adam(model.parameters(), lr=1e-2).step()
    assert model.fusion.p_v.item() != 1.0 and model.fusion.p_a.item() != 1.0


def test_determinism(kind):
    def run():
        torch.manual_seed(123)
        model = small_multimodal(kind).eval()
        return model(*small_inputs())

    assert torch.equal(run(), run())
