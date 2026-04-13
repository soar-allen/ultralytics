"""
CVAT 双向同步模块。

严格映射关系：
  FiftyOne Dataset  ↔  CVAT Project
  FiftyOne View     →  CVAT Task
  CVAT Job          ↔  Task 的自动/手动切分批次

支持的标注类型：分类、检测框、多边形、关键点（skeleton）。
使用 fiftyone.utils.cvat 进行交互，必要时回退到 CVAT REST API。
"""

from __future__ import annotations

import logging
from typing import Optional

import fiftyone as fo
import fiftyone.utils.cvat as fouc

from .config import CONFIG

logger = logging.getLogger(__name__)


# ===================================================================
# 连接配置
# ===================================================================

def _get_cvat_cred_kwargs() -> dict:
    """仅含认证信息的 CVAT 连接参数（不含 organization），用于 load_annotations 等接口。"""
    return {
        "url": CONFIG.cvat.url,
        "username": CONFIG.cvat.username,
        "password": CONFIG.cvat.password,
    }


def _get_cvat_kwargs() -> dict:
    """构建 CVAT 连接参数。包含 organization 时推送/拉取都会指向该组织空间。"""
    kwargs = _get_cvat_cred_kwargs()
    org = (CONFIG.cvat.organization or "").strip()
    if org:
        kwargs["organization"] = org
    return kwargs


# ===================================================================
# 连接测试
# ===================================================================

def test_connection() -> dict:
    """
    测试 CVAT 连接，返回连接状态和服务器信息。

    Returns:
        {
            "connected": bool,
            "url": str,
            "username": str,
            "server_version": str | None,
            "projects_count": int | None,
            "tasks_count": int | None,
            "error": str | None,
            "solution": str | None,
        }
    """
    result = {
        "connected": False,
        "url": CONFIG.cvat.url,
        "username": CONFIG.cvat.username,
        "server_version": None,
        "projects_count": None,
        "tasks_count": None,
        "error": None,
        "solution": None,
    }

    if not CONFIG.cvat.url:
        result["error"] = "CVAT URL 未配置"
        result["solution"] = "请在侧边栏填写 CVAT 服务器地址（如 http://localhost:8080）"
        return result

    if not CONFIG.cvat.username or not CONFIG.cvat.password:
        result["error"] = "CVAT 用户名或密码未配置"
        result["solution"] = "请在侧边栏填写 CVAT 登录凭据"
        return result

    try:
        import requests
        url = CONFIG.cvat.url.rstrip("/")

        resp = requests.get(
            f"{url}/api/server/about",
            auth=(CONFIG.cvat.username, CONFIG.cvat.password),
            timeout=10,
        )

        if resp.status_code == 401:
            result["error"] = "认证失败：用户名或密码错误"
            result["solution"] = "请检查侧边栏中的 CVAT 用户名和密码是否正确"
            return result
        elif resp.status_code == 404:
            about = None
        else:
            resp.raise_for_status()
            about = resp.json()

        if about:
            result["server_version"] = about.get("version", "未知")

        proj_resp = requests.get(
            f"{url}/api/projects?page_size=1",
            auth=(CONFIG.cvat.username, CONFIG.cvat.password),
            timeout=10,
        )
        if proj_resp.ok:
            result["projects_count"] = proj_resp.json().get("count", 0)

        task_resp = requests.get(
            f"{url}/api/tasks?page_size=1",
            auth=(CONFIG.cvat.username, CONFIG.cvat.password),
            timeout=10,
        )
        if task_resp.ok:
            result["tasks_count"] = task_resp.json().get("count", 0)

        org_resp = requests.get(
            f"{url}/api/organizations",
            auth=(CONFIG.cvat.username, CONFIG.cvat.password),
            timeout=10,
        )
        if org_resp.ok:
            orgs = org_resp.json().get("results", org_resp.json() if isinstance(org_resp.json(), list) else [])
            result["organizations"] = [
                {"slug": o.get("slug", ""), "name": o.get("name", "")} for o in orgs
            ]

        result["connected"] = True

    except requests.exceptions.ConnectionError:
        result["error"] = f"无法连接到 CVAT 服务器: {CONFIG.cvat.url}"
        result["solution"] = (
            "请检查：\n"
            "1. CVAT 服务器是否已启动\n"
            "2. URL 地址是否正确（含端口号）\n"
            "3. 网络是否可达（防火墙/VPN）\n"
            "4. 如果使用 Docker，确认端口映射正确"
        )
    except requests.exceptions.Timeout:
        result["error"] = "连接超时"
        result["solution"] = "CVAT 服务器响应缓慢，请检查服务器负载或网络状况"
    except Exception as e:
        result["error"] = str(e)
        result["solution"] = "请检查 CVAT 服务器配置和网络连接"

    return result


# ===================================================================
# 推送到 CVAT
# ===================================================================

OCCLUSION_ATTRS = {
    "1_occu": {"type": "checkbox", "values": [True, False], "default": False},
    "2_occu": {"type": "checkbox", "values": [True, False], "default": False},
    "3_occu": {"type": "checkbox", "values": [True, False], "default": False},
    "4_occu": {"type": "checkbox", "values": [True, False], "default": False},
}


def push_to_cvat(
    samples: fo.Dataset | fo.DatasetView,
    anno_key: str,
    label_field: Optional[str] = None,
    project_name: Optional[str] = None,
    task_name: Optional[str] = None,
    segment_size: int = 200,
    image_quality: int = 100,
    label_type: Optional[str] = None,
    classes: Optional[list[str]] = None,
    occluded_attr: Optional[str] = None,
    attributes: Optional[dict] = None,
    label_schema: Optional[dict] = None,
) -> dict:
    """
    将数据集或视图推送到 CVAT 生成标注任务。

    支持两种模式：
      1. 单字段模式：指定 label_field（+ label_type / classes / attributes）
      2. 多字段模式：指定 label_schema，一次推送多种标注类型到同一 CVAT 任务

    label_schema 格式示例::

        {
            "ground_truth": {                          # 已有字段，自动检测类型
                "classes": ["classA"],                  # 可选，限制类别
            },
            "ground_truth_polylines": {
                "attributes": {                        # 自定义属性
                    "1_occu": {"type": "checkbox", "values": [True, False], "default": False},
                },
            },
            "new_field": {                             # 新字段，必须指定 type
                "type": "detections",
                "classes": ["X", "Y"],
            },
        }

    当 label_schema 提供时，label_field / label_type / classes / attributes 被忽略。
    """
    cvat_kwargs = _get_cvat_kwargs()
    ds = samples if isinstance(samples, fo.Dataset) else samples._dataset

    if project_name is None:
        project_name = ds.name

    if task_name is None:
        task_name = f"{ds.name}_{anno_key}"

    kwargs = {
        "project_name": project_name,
        "task_name": task_name,
        "segment_size": segment_size,
        "image_quality": image_quality,
        **cvat_kwargs,
    }

    if label_schema:
        kwargs["label_schema"] = label_schema
        pushed_fields = list(label_schema.keys())
    else:
        if label_field is None:
            label_field = "ground_truth"
        field_exists = label_field in ds.get_field_schema()
        kwargs["label_field"] = label_field
        if label_type and not field_exists:
            kwargs["label_type"] = label_type
        if classes:
            kwargs["classes"] = classes
        elif field_exists:
            from .data_manager import get_label_classes
            detected_cls = get_label_classes(ds, label_field)
            if detected_cls:
                kwargs["classes"] = detected_cls
        if occluded_attr:
            kwargs["occluded_attr"] = occluded_attr
        if attributes:
            kwargs["attributes"] = attributes
        pushed_fields = [label_field]

    logger.info(
        "推送到 CVAT: project=%s, task=%s, fields=%s, samples=%d",
        project_name, task_name, pushed_fields, len(samples),
    )

    try:
        samples.annotate(anno_key, **kwargs)
    except Exception as e:
        # annotate() 在调用 CVAT API 前就会注册 anno_key，
        # 失败后需要清理，否则用户无法用同一 key 重试
        try:
            if anno_key in ds.list_annotation_runs():
                ds.delete_annotation_run(anno_key)
                logger.info("推送失败，已自动清理残留 anno_key: %s", anno_key)
        except Exception:
            pass

        err_msg = str(e).lower()
        if "connection" in err_msg or "connect" in err_msg:
            raise ConnectionError(
                f"无法连接 CVAT 服务器 ({CONFIG.cvat.url})。\n"
                "解决办法：\n"
                "1. 确认 CVAT 服务已启动\n"
                "2. 检查 URL 和端口是否正确\n"
                "3. 使用「连接测试」按钮验证连通性"
            ) from e
        if "401" in err_msg or "auth" in err_msg or "credential" in err_msg:
            raise PermissionError(
                "CVAT 认证失败。\n"
                "解决办法：检查侧边栏中的用户名和密码"
            ) from e
        if "same organization" in err_msg:
            org = (CONFIG.cvat.organization or "").strip()
            if org:
                raise ValueError(
                    f"CVAT 组织不匹配：当前设置的组织为 '{org}'，但任务与项目不在同一组织下。\n"
                    "解决办法：在侧边栏「CVAT 配置 → Organization」中检查组织 slug 是否正确，"
                    "或留空以推送到个人空间。可通过「测试 CVAT 连接」查看你的账号所属组织。"
                ) from e
            else:
                raise ValueError(
                    "CVAT 组织不匹配：当前未设置组织，但 CVAT 端存在同名项目属于某个组织。\n"
                    "解决办法：\n"
                    "1. 在侧边栏「CVAT 配置 → Organization」中填写正确的组织 slug\n"
                    "2. 或使用不同的项目名称推送到个人空间\n"
                    "可通过「测试 CVAT 连接」查看你的账号所属组织。"
                ) from e
        if "already exists" in err_msg or "anno_key" in err_msg:
            raise ValueError(
                f"标注键 '{anno_key}' 已存在。\n"
                "解决办法：使用不同的 anno_key，或在「管理标注运行」中删除旧记录"
            ) from e
        raise

    info = {
        "anno_key": anno_key,
        "project_name": project_name,
        "task_name": task_name,
        "label_fields": pushed_fields,
        "num_samples": len(samples),
    }
    logger.info("推送成功: %s", info)
    return info


def push_detections_to_cvat(
    samples: fo.Dataset | fo.DatasetView,
    anno_key: str,
    label_field: str = "ground_truth",
    **kwargs,
) -> dict:
    """推送检测框标注到 CVAT。"""
    return push_to_cvat(
        samples, anno_key, label_field=label_field,
        label_type="detections", **kwargs,
    )


def push_polylines_to_cvat(
    samples: fo.Dataset | fo.DatasetView,
    anno_key: str,
    label_field: str = "ground_truth_polylines",
    include_occlusion_attrs: bool = False,
    **kwargs,
) -> dict:
    """推送多边形标注到 CVAT。默认附带四角遮挡 checkbox 属性。"""
    if include_occlusion_attrs and "attributes" not in kwargs:
        kwargs["attributes"] = OCCLUSION_ATTRS
    return push_to_cvat(
        samples, anno_key, label_field=label_field,
        label_type="polylines", **kwargs,
    )


def push_keypoints_to_cvat(
    samples: fo.Dataset | fo.DatasetView,
    anno_key: str,
    label_field: str = "ground_truth_keypoints",
    **kwargs,
) -> dict:
    """推送关键点标注到 CVAT。"""
    return push_to_cvat(
        samples, anno_key, label_field=label_field,
        label_type="keypoints", **kwargs,
    )


# ===================================================================
# 从 CVAT 拉取
# ===================================================================

def pull_from_cvat(
    samples: fo.Dataset | fo.DatasetView,
    anno_key: str,
    cleanup: bool = False,
) -> dict:
    """
    从 CVAT 拉取标注结果。

    Args:
        samples: 关联的 FiftyOne Dataset 或 DatasetView
        anno_key: 之前推送时使用的标注键
        cleanup: 是否在拉取后删除 CVAT 端的任务
    """
    ds = samples if isinstance(samples, fo.Dataset) else samples._dataset
    logger.info("从 CVAT 拉取标注: anno_key=%s", anno_key)

    available_runs = ds.list_annotation_runs()
    if anno_key not in available_runs:
        raise ValueError(
            f"标注键 '{anno_key}' 不存在于当前数据集中。\n"
            f"可用的标注运行: {', '.join(available_runs) if available_runs else '无'}\n"
            "解决办法：先通过「推送到 CVAT」创建标注任务，然后再拉取"
        )

    try:
        ds.load_annotations(anno_key, cleanup=cleanup, **_get_cvat_cred_kwargs())
    except Exception as e:
        err_msg = str(e).lower()
        if "connection" in err_msg or "connect" in err_msg:
            raise ConnectionError(
                f"无法连接 CVAT 服务器 ({CONFIG.cvat.url})。\n"
                "解决办法：\n"
                "1. 确认 CVAT 服务已启动\n"
                "2. 检查 URL 和端口是否正确\n"
                "3. 使用「连接测试」按钮验证连通性"
            ) from e
        if "401" in err_msg or "auth" in err_msg:
            raise PermissionError(
                "CVAT 认证失败。\n"
                "解决办法：检查侧边栏中的用户名和密码"
            ) from e
        if "404" in err_msg or "not found" in err_msg:
            raise ValueError(
                f"CVAT 端的标注任务不存在（可能已被删除）。\n"
                "解决办法：\n"
                "1. 在「管理标注运行」中删除此 anno_key 记录\n"
                "2. 重新推送数据并创建新的标注任务"
            ) from e
        raise

    info = {
        "anno_key": anno_key,
        "status": "loaded",
    }
    logger.info("拉取完成: %s", info)
    return info


# ===================================================================
# 标注管理
# ===================================================================

def list_annotation_runs(ds: fo.Dataset) -> list[dict]:
    """列出数据集上的所有标注运行。"""
    runs = []
    for key in ds.list_annotation_runs():
        info = ds.get_annotation_info(key)
        runs.append({
            "anno_key": key,
            "timestamp": str(info.timestamp) if hasattr(info, "timestamp") else "N/A",
            "config": str(info.config) if hasattr(info, "config") else "N/A",
        })
    return runs


def delete_annotation_run(ds: fo.Dataset, anno_key: str, cleanup: bool = False) -> None:
    """删除标注运行记录。cleanup=True 时同时删除 CVAT 端的任务。"""
    if cleanup:
        try:
            results = ds.load_annotation_results(anno_key, **_get_cvat_cred_kwargs())
            results.cleanup()
        except Exception as e:
            logger.warning("CVAT 清理失败 (anno_key=%s): %s", anno_key, e)

    ds.delete_annotation_run(anno_key)
    logger.info("已删除标注运行: %s", anno_key)


def get_annotation_status(ds: fo.Dataset, anno_key: str) -> dict:
    """查询 CVAT 标注任务状态。"""
    try:
        results = ds.load_annotation_results(anno_key, **_get_cvat_cred_kwargs())
        api = results.connect_to_api()

        status_info = {"anno_key": anno_key, "tasks": []}
        task_ids = results.task_ids if hasattr(results, "task_ids") else []
        for tid in task_ids:
            try:
                task_data = api.get(f"/api/tasks/{tid}")
                status_info["tasks"].append({
                    "task_id": tid,
                    "name": task_data.get("name", ""),
                    "status": task_data.get("status", "unknown"),
                    "size": task_data.get("size", 0),
                })
            except Exception:
                status_info["tasks"].append({"task_id": tid, "status": "error"})

        return status_info
    except Exception as e:
        return {"anno_key": anno_key, "error": str(e)}


# ===================================================================
# Job 级别状态查询
# ===================================================================

REVIEW_FIELD = "_review"
REVIEW_CLASSES = ["discard"]


def _cvat_get_paginated(endpoint: str, params: Optional[dict] = None) -> list:
    """从 CVAT REST API 分页获取所有结果。"""
    import requests

    base = CONFIG.cvat.url.rstrip("/")
    auth = (CONFIG.cvat.username, CONFIG.cvat.password)
    results: list = []
    page_url = f"{base}{endpoint}"

    while page_url:
        resp = requests.get(page_url, auth=auth, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            results.extend(data)
            break
        results.extend(data.get("results", []))
        page_url = data.get("next")
        params = None
    return results


def get_job_details(ds: fo.Dataset, anno_key: str) -> list[dict]:
    """查询某个标注运行下所有 CVAT Job 的详细状态。

    CVAT Job 生命周期::

        stage (工作阶段): annotation → validation → acceptance
        state (完成状态): new → in progress → completed / rejected

    Returns:
        每个 Job 的信息列表，包含 job_id / task_id / state / stage /
        assignee / frame 范围 / 关联的 FiftyOne sample_ids 等。
    """
    results = ds.load_annotation_results(anno_key, **_get_cvat_cred_kwargs())
    task_ids = getattr(results, "task_ids", [])
    frame_id_map = getattr(results, "frame_id_map", {})

    all_jobs: list[dict] = []
    for tid in task_ids:
        try:
            jobs = _cvat_get_paginated("/api/jobs", params={"task_id": tid, "page_size": 100})
        except Exception as e:
            logger.warning("获取 task %s 的 jobs 失败: %s", tid, e)
            continue

        task_frames: dict = frame_id_map.get(tid, frame_id_map.get(str(tid), {}))

        for job in jobs:
            start = job.get("start_frame", 0)
            stop = job.get("stop_frame", 0)

            sample_ids: list[str] = []
            for fid, sid in task_frames.items():
                fid_int = int(fid) if isinstance(fid, str) else fid
                if start <= fid_int <= stop:
                    sample_ids.append(sid)

            all_jobs.append({
                "job_id": job["id"],
                "task_id": tid,
                "state": job.get("state", "unknown"),
                "stage": job.get("stage", "unknown"),
                "assignee": (job.get("assignee") or {}).get("username", ""),
                "start_frame": start,
                "stop_frame": stop,
                "frame_count": stop - start + 1,
                "num_samples": len(sample_ids) if sample_ids else (stop - start + 1),
                "sample_ids": sample_ids,
            })
    return all_jobs


def tag_samples_by_job_status(
    ds: fo.Dataset,
    anno_key: str,
    tag_prefix: str = "job",
) -> dict[str, int]:
    """根据 CVAT Job 状态为 FiftyOne 样本打标签。

    为每个样本添加 ``{prefix}_{state}`` 和 ``{prefix}_{stage}`` 标签，
    例如 ``job_completed``、``job_acceptance``。
    """
    jobs = get_job_details(ds, anno_key)
    tagged: dict[str, int] = {}

    for job in jobs:
        sids = job["sample_ids"]
        if not sids:
            continue
        state_tag = f"{tag_prefix}_{job['state'].replace(' ', '_')}"
        stage_tag = f"{tag_prefix}_{job['stage'].replace(' ', '_')}"
        view = ds.select(sids)
        for tag in (state_tag, stage_tag):
            view.tag_samples(tag)
            tagged[tag] = tagged.get(tag, 0) + len(sids)
    return tagged


def get_samples_by_job_status(
    ds: fo.Dataset,
    anno_key: str,
    states: Optional[list[str]] = None,
    stages: Optional[list[str]] = None,
) -> fo.DatasetView:
    """获取特定 Job 状态/阶段下的样本视图。

    *states* 与 *stages* 为 AND 关系，``None`` 表示不过滤该维度。
    """
    jobs = get_job_details(ds, anno_key)
    matching_ids: list[str] = []
    for job in jobs:
        state_ok = states is None or job["state"] in states
        stage_ok = stages is None or job["stage"] in stages
        if state_ok and stage_ok:
            matching_ids.extend(job["sample_ids"])
    if not matching_ids:
        return ds.limit(0)
    return ds.select(matching_ids)


# ===================================================================
# 废弃图片管理
# ===================================================================

def find_discarded_samples(
    ds: fo.Dataset,
    review_field: str = REVIEW_FIELD,
) -> fo.DatasetView:
    """查找在 CVAT 中被标记为「废弃」的样本。

    推送时需启用废弃标记字段，标注员在 CVAT 的 tag 面板选择 ``discard``，
    拉取后调用本函数即可获取对应样本视图。
    """
    from fiftyone import ViewField as F

    schema = ds.get_field_schema()
    if review_field not in schema:
        return ds.limit(0)
    return ds.filter_labels(review_field, F("label") == "discard", only_matches=True)


def handle_discarded_samples(
    ds: fo.Dataset,
    review_field: str = REVIEW_FIELD,
    action: str = "tag",
    tag_name: str = "discarded",
) -> dict:
    """处理废弃样本。

    Args:
        action:
            - ``"tag"``: 为废弃样本打上 *tag_name* 标签
            - ``"remove"``: 从数据集中移除（保留物理文件）
            - ``"remove_and_delete"``: 移除并删除物理文件
    """
    discarded = find_discarded_samples(ds, review_field)
    count = len(discarded)
    if count == 0:
        return {"action": action, "count": 0, "message": "未发现废弃样本"}

    if action == "tag":
        discarded.tag_samples(tag_name)
        return {"action": action, "count": count, "tag": tag_name}

    if action in ("remove", "remove_and_delete"):
        sample_ids = discarded.values("id")
        filepaths = discarded.values("filepath") if action == "remove_and_delete" else []
        ds.delete_samples(sample_ids)

        deleted_files = 0
        if filepaths:
            import os
            for p in filepaths:
                try:
                    if os.path.isfile(p):
                        os.remove(p)
                        deleted_files += 1
                except OSError as e:
                    logger.warning("删除文件失败 %s: %s", p, e)

        result: dict = {"action": action, "count": count}
        if deleted_files:
            result["deleted_files"] = deleted_files
        return result

    raise ValueError(f"未知操作: {action}")
