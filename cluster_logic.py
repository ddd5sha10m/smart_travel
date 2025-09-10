# cluster_logic.py (智慧錨定分群法)

import numpy as np

def perform_clustering(time_matrix_seconds):
    """
    使用基於「智慧錨點」(最高連接度)的演算法對地點進行分群。
    """
    num_locations = len(time_matrix_seconds)
    if num_locations == 0:
        return []

    # 步行半徑設為 15 分鐘 (900秒)
    threshold_seconds = 15 * 60
    unassigned_indices = list(range(num_locations))
    final_clusters = []

    print("開始使用智慧錨點分群法...")

    while unassigned_indices:
        best_anchor_idx = -1
        max_connections = -1

        # 1. 對每一個尚未分群的點，計算它的「連接數」(朋友圈大小)
        for candidate_idx in unassigned_indices:
            current_connections = 0
            for other_idx in unassigned_indices:
                if candidate_idx == other_idx:
                    continue
                if time_matrix_seconds[candidate_idx][other_idx] <= threshold_seconds:
                    current_connections += 1
            
            if current_connections > max_connections:
                max_connections = current_connections
                best_anchor_idx = candidate_idx

        # 如果所有剩下的點都是孤立的，就選第一個
        if best_anchor_idx == -1:
            best_anchor_idx = unassigned_indices[0]
        
        anchor_point_idx = best_anchor_idx

        new_cluster = []
        to_be_assigned = []
        new_cluster.append(anchor_point_idx)
        to_be_assigned.append(anchor_point_idx)

        # 遍歷所有未分配的點，看它們是否靠近新的「人氣王」錨點
        for candidate_idx in unassigned_indices:
            if candidate_idx == anchor_point_idx:
                continue
            if time_matrix_seconds[anchor_point_idx][candidate_idx] <= threshold_seconds:
                new_cluster.append(candidate_idx)
                to_be_assigned.append(candidate_idx)

        unassigned_indices = [idx for idx in unassigned_indices if idx not in to_be_assigned]
        
        final_clusters.append(new_cluster)
        print(f"  - 建立新群集 (智慧錨點: {anchor_point_idx}, 連接數: {max_connections}): {new_cluster}")

    print(f"智慧錨點分群結果 (地點索引): {final_clusters}")
    return final_clusters