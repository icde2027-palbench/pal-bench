from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.benchmark.dataset_builder import BenchmarkDatasetBuilder, audit_dual_view, build_dual_benchmark_views
from src.llm import create_llm_for_role
from src.utils.io import load_json, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _iter_user_dirs(base: Path, user_id: str | None):
    if user_id:
        yield base / user_id
    else:
        for d in sorted(base.iterdir()):
            if d.is_dir() and re.fullmatch(r"user_\d{4}", d.name):
                yield d


def run(
    config_path: str,
    data_dir: str,
    output_dir: str,
    user_id: str | None,
    resume: bool,
    backend_name: str,
) -> None:
    cfg = _load_config(config_path)
    bench_cfg = cfg.get("benchmark", {}) or {}
    entity_role = str(bench_cfg.get("entity_llm_role") or "agent_llm")
    llm = create_llm_for_role(entity_role)
    batch_size = int(bench_cfg.get("description_batch_size", 1) or 1)
    text_entity_workers = int(bench_cfg.get("text_entity_max_workers", 8) or 8)
    text_entity_retries = int(bench_cfg.get("text_entity_max_retries", 2) or 2)
    album_target_cfg = bench_cfg.get("album_photo_target") or {}
    min_total = int(album_target_cfg.get("min_total") or 500)
    max_total = int(album_target_cfg.get("max_total") or 1000)
    builder = BenchmarkDatasetBuilder(
        llm=llm,
        description_batch_size=batch_size,
        target_album_photo_min=min_total,
        target_album_photo_max=max_total,
        text_entity_max_workers=text_entity_workers,
        text_entity_max_retries=text_entity_retries,
    )

    data_base = Path(data_dir)
    out_base = Path(output_dir)
    user_dirs = list(_iter_user_dirs(data_base, user_id))
    total = len(user_dirs)
    done = 0

    for index, user_dir in enumerate(user_dirs, 1):
        uid = user_dir.name
        out_user_dir = out_base / uid
        agent_file = out_user_dir / f"{uid}_agent_album.json"
        gt_file = out_user_dir / f"{uid}_eval_gt.json"
        audit_file = out_user_dir / f"{uid}_export_audit.json"

        required = [
            user_dir / f"{uid}.json",
            user_dir / f"{uid}_social_graph.json",
            user_dir / f"{uid}_reasoning_paths.json",
        ]
        timeline_file = user_dir / f"{uid}_adjusted_timeline.json"
        if not timeline_file.exists():
            timeline_file = user_dir / f"{uid}_timeline.json"
        required.append(timeline_file)
        missing = [path.name for path in required if not path.exists()]
        if missing:
            logger.warning("[%d/%d] %s 缺少依赖文件，跳过: %s", index, total, uid, missing)
            continue

        if resume and agent_file.exists() and gt_file.exists():
            try:
                existing_audit = audit_dual_view(load_json(agent_file), load_json(gt_file))
                if existing_audit.get("passed"):
                    logger.info("[%d/%d] %s 双视图 benchmark 已存在且 audit 通过，跳过", index, total, uid)
                    continue
                logger.warning("[%d/%d] %s 现有双视图 audit 未通过，将重建", index, total, uid)
            except Exception as exc:
                logger.warning("[%d/%d] %s 现有双视图读取/audit 失败，将重建: %s", index, total, uid, exc)

        logger.info("[%d/%d] 导出 %s 的 benchmark 双视图数据 ...", index, total, uid)
        try:
            result = builder.build_from_user_dir(
                user_dir,
                preferred_backend=backend_name,
                save_ambient_plan=True,
            )
            agent_album, eval_gt, audit = build_dual_benchmark_views(result)
            if not audit.get("passed"):
                raise RuntimeError(f"dual-view export audit failed: {audit.get('errors', [])[:5]}")
            out_user_dir.mkdir(parents=True, exist_ok=True)
            save_json(agent_album, agent_file)
            save_json(eval_gt, gt_file)
            save_json(audit, audit_file)
            logger.info(
                "  ✓ %s 双视图已保存（%d 张照片，%d 个 eval targets）",
                uid,
                agent_album.get("album_summary", {}).get("n_photos", 0),
                len(eval_gt.get("evaluation_targets", [])),
            )
            done += 1
        except Exception as exc:
            logger.error("  ✗ %s benchmark 导出失败：%s", uid, exc, exc_info=True)

    logger.info("完成：共导出 %d 个用户的 benchmark 数据", done)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 数据导出")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--data", default="data/full/users")
    parser.add_argument("--output", default="data/full/users")
    parser.add_argument("--user_id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--backend", default="gemini",
                        help="优先读取的 manifest 后端（默认 gemini）")
    args = parser.parse_args()

    run(
        config_path=args.config,
        data_dir=args.data,
        output_dir=args.output,
        user_id=args.user_id,
        resume=args.resume,
        backend_name=args.backend,
    )


if __name__ == "__main__":
    main()
