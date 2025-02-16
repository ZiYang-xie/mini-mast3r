import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path
from argparse import ArgumentParser
import torch
import os

from mini_mast3r.api import OptimizedResult, inferece_mast3r, log_optimized_result
from mini_mast3r.model import AsymmetricMASt3R


def create_blueprint(image_name_list: list[str], log_path: Path) -> rrb.Blueprint:
    # dont show 2d views if there are more than 4 images as to not clutter the view
    if len(image_name_list) > 4:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin=f"{log_path}"),
            ),
            collapse_panels=True,
        )
    else:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                contents=[
                    rrb.Spatial3DView(origin=f"{log_path}"),
                    rrb.Vertical(
                        contents=[
                            rrb.Spatial2DView(
                                origin=f"{log_path}/camera_{i}/pinhole/",
                                contents=[
                                    "+ $origin/**",
                                ],
                            )
                            for i in range(len(image_name_list))
                        ]
                    ),
                ],
                column_shares=[3, 1],
            ),
            collapse_panels=True,
        )
    return blueprint


def main(image_dir: Path):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    ).to(device)

    optimized_results: OptimizedResult = inferece_mast3r(
        image_dir_or_list=image_dir,
        model=model,
        device=device,
        batch_size=1,
        match_mode="swin-7",
        max_num=20
    )

    image_dir = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    blueprint = create_blueprint(image_dir, "world")
    rr.send_blueprint(blueprint)
    log_optimized_result(optimized_results, Path("world"))


if __name__ == "__main__":
    parser = ArgumentParser("mini-dust3r rerun demo script")
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing images to process",
        required=True,
    )
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "mini-dust3r")
    main(args.image_dir)
    rr.script_teardown(args)
