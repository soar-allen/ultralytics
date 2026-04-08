#!/usr/bin/env python3
"""
replace_by_name.py - 用源目录中的文件替换目标目录中的同名文件

用法:
    python replace_by_name.py <源目录> <目标目录>
    python replace_by_name.py --help
"""

import os
import sys
import shutil
import argparse

def build_source_map(source_dir):
    """
    遍历源目录，构建文件名 -> 文件完整路径 的映射。
    如果出现重复文件名，输出警告并只保留第一个。
    """
    source_map = {}
    for root, _, files in os.walk(source_dir):
        for file in files:
            full_path = os.path.join(root, file)
            if file in source_map:
                print(f"警告: 源目录中存在重复文件名 '{file}'，"
                      f"已使用: {source_map[file]}，忽略: {full_path}",
                      file=sys.stderr)
            else:
                source_map[file] = full_path
    return source_map

def replace_and_report(target_dir, source_map, verbose=False):
    """
    遍历目标目录，对每个文件尝试用源目录中的同名文件替换。
    返回 (总文件数, 成功替换数, 失败列表)
    失败列表中每个元素为 (文件路径, 失败原因)
    """
    total = 0
    success = 0
    failed = []

    for root, _, files in os.walk(target_dir):
        for file in files:
            total += 1
            target_path = os.path.join(root, file)

            if file not in source_map:
                failed.append((target_path, "源目录中无同名文件"))
                continue

            source_path = source_map[file]
            try:
                shutil.copy2(source_path, target_path)   # 复制并尽量保留元数据
                success += 1
                if verbose:
                    print(f"成功替换: {target_path} <- {source_path}")
            except Exception as e:
                failed.append((target_path, f"复制失败: {e}"))

    return total, success, failed

def main():
    parser = argparse.ArgumentParser(
        description="将源目录中与目标目录同名的文件替换掉目标目录中的文件"
    )
    parser.add_argument("source_dir", help="第一个目录（源目录）")
    parser.add_argument("target_dir", help="第二个目录（目标目录）")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="显示详细的替换信息")
    args = parser.parse_args()

    # 检查目录有效性
    if not os.path.isdir(args.source_dir):
        print(f"错误: 源目录 '{args.source_dir}' 不存在或不是目录", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.target_dir):
        print(f"错误: 目标目录 '{args.target_dir}' 不存在或不是目录", file=sys.stderr)
        sys.exit(1)

    # 构建源文件映射表
    print(f"正在扫描源目录: {args.source_dir}")
    source_map = build_source_map(args.source_dir)
    print(f"源目录中共找到 {len(source_map)} 个唯一文件名")

    # 执行替换并统计
    print(f"正在处理目标目录: {args.target_dir}")
    total, success, failed = replace_and_report(
        args.target_dir, source_map, args.verbose
    )

    # 输出最终统计
    print("\n===== 统计结果 =====")
    print(f"目标目录总文件数: {total}")
    print(f"成功替换文件数: {success}")
    print(f"未替换文件数: {len(failed)}")

    if failed:
        print("\n未替换的文件列表:")
        for file_path, reason in failed:
            print(f"  - {file_path} (原因: {reason})")
    else:
        print("所有文件均已成功替换。")

    # 如果有未替换的文件，返回非零退出码
    sys.exit(0 if len(failed) == 0 else 1)

if __name__ == "__main__":
    main()