"""Figure captions are derived from the data they describe."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hddp import figures  # noqa: E402


def test_auto_algorithm_switch_is_read_from_verification_logs():
    share = figures.auto_tree_share()
    assert share[("vit_b_16", 128.0)] == 0.7  # 7 of 10 logged gradient all-reduces used Tree
    assert all(f < 0.07 for (m, b), f in share.items() if (m, b) != ("vit_b_16", 128.0))
    fig, ax = plt.subplots()
    note = figures.mark_algorithm_switch(ax, "vit_b_16", "4gpu", "Auto")
    assert "at 128 MB NCCL chose Tree for 70%" in note
    assert figures.mark_algorithm_switch(ax, "resnet50", "4gpu", "Auto") == ""
    assert figures.mark_algorithm_switch(ax, "vit_b_16", "4gpu", "Tree") == ""
    plt.close(fig)


def test_error_bar_caption_reflects_sample_size():
    assert figures.ci_note([5, 4, 5]) == "Error bars: 95% t-CI over independent launches (n = 4-5)."
    assert "Points with n = 1 have no error bar." in figures.ci_note([1, 5])
    assert figures.ci_note([None]) == ""


def test_non_nvlink_setting_is_not_labelled_pcie_p2p():
    assert "SHM" in figures.SET_LABEL["2gpu_pcie"] and "PCIe" not in figures.SET_LABEL["2gpu_pcie"]
    assert "SHM" in figures.SET_LABEL["4gpu"]
