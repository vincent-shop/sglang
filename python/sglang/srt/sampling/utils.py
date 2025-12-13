from sglang.srt.utils import is_cuda

try:
    from flashinfer.sampling import min_p_sampling_from_probs
    from flashinfer.sampling import top_k_renorm_probs as top_k_renorm_prob
    from flashinfer.sampling import top_k_top_p_sampling_from_probs
    from flashinfer.sampling import top_p_renorm_probs as top_p_renorm_prob

    # Check for raw top_k API (added in PR #2119)
    try:
        from flashinfer import top_k as top_k_raw
    except ImportError:
        top_k_raw = None

    flashinfer_available = True
except ImportError:
    flashinfer_available = False
    top_k_raw = None

if not flashinfer_available and is_cuda():
    from sgl_kernel import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )

    # sgl_kernel might not have raw top_k
    top_k_raw = None


def get_top_k_renorm_prob():
    return top_k_renorm_prob


def get_top_p_renorm_prob():
    return top_p_renorm_prob


def get_min_p_sampling_from_probs():
    return min_p_sampling_from_probs


def get_top_k_top_p_sampling_from_probs():
    return top_k_top_p_sampling_from_probs


def get_top_k_raw():
    return top_k_raw
