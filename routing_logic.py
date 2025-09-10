
# routing_logic.py (重構與優化版本)

import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import itertools

UNREACHABLE = 9999999
TRANSITION_PENALTY_SECONDS = 12 * 60 # 跨區移動的隱藏時間成本

# --- 輔助函式 ---
def find_closest_point_pair(gmaps_client, origins_addrs, dests_addrs, mode='driving'):
    min_duration = float('inf')
    best_origin_idx, best_dest_idx = -1, -1
    try:
        matrix_result = gmaps_client.distance_matrix(origins=origins_addrs, destinations=dests_addrs, mode=mode)
        for i, row in enumerate(matrix_result['rows']):
            for j, element in enumerate(row['elements']):
                if element['status'] == 'OK':
                    duration = element['duration']['value']
                    if duration < min_duration:
                        min_duration = duration
                        best_origin_idx = i
                        best_dest_idx = j
    except Exception as e:
        print(f"find_closest_point_pair 出錯: {e}")
    return min_duration, best_origin_idx, best_dest_idx

def calculate_cluster_matrix(gmaps_client, locations_list, clusters, mode='driving'):
    num_clusters = len(clusters)
    cluster_matrix = np.full((num_clusters, num_clusters), UNREACHABLE, dtype=int)
    cluster_addresses = [[locations_list[i] for i in cluster] for cluster in clusters]
    for i in range(num_clusters):
        for j in range(num_clusters):
            if i == j:
                cluster_matrix[i][j] = 0
                continue
            duration, _, _ = find_closest_point_pair(gmaps_client, cluster_addresses[i], cluster_addresses[j], mode=mode)
            if duration != float('inf'):
                # 跨區移動加上轉換成本
                cluster_matrix[i][j] = duration + TRANSITION_PENALTY_SECONDS
    return cluster_matrix

# --- 演算法 ---
def solve_cluster_tsp_bruteforce(cluster_matrix):
    num_clusters = len(cluster_matrix)
    if num_clusters <= 2:
        return list(range(num_clusters))
    min_path_duration = float('inf')
    best_path = []
    for permutation in itertools.permutations(range(num_clusters)):
        current_duration = sum(cluster_matrix[permutation[i]][permutation[i+1]] for i in range(num_clusters - 1))
        if current_duration < min_path_duration:
            min_path_duration = current_duration
            best_path = list(permutation)
    print(f"窮舉法找到的最佳群集順序: {best_path}")
    return best_path

def solve_points_tsp_with_ga(point_indices, full_walking_matrix):
    num_points = len(point_indices)
    if num_points <= 2:
        return point_indices

    sub_matrix = np.array([[full_walking_matrix[i][j] for j in point_indices] for i in point_indices])
    index_map = {i: original_idx for i, original_idx in enumerate(point_indices)}

    def fitness_function(solution):
        if len(set(solution)) < num_points:
            return float('inf')
        total_time = sum(sub_matrix[int(solution[i])][int(solution[i+1])] for i in range(num_points - 1))
        return total_time

    varbound = np.array([[0, num_points - 1]] * num_points)
    algorithm_param = {
        'max_num_iteration': 1500, 'population_size': 100, 'mutation_probability': 0.1,
        'elit_ratio': 0.01, 'crossover_probability': 0.5, 'parents_portion': 0.3,
        'crossover_type': 'one_point', 'max_iteration_without_improv': None
    }
    model = ga(function=fitness_function, dimension=num_points, variable_type='int', variable_boundaries=varbound, algorithm_parameters=algorithm_param, convergence_curve=False, progress_bar=False)
    model.run()
    
    best_local_route = model.best_variable.astype(int)
    return [index_map[i] for i in best_local_route]

# --- 核心拼接邏輯 ---
def create_final_itinerary(locations, clusters, cluster_order, walking_matrix, gmaps_client, mode='driving'):
    final_route_indices = []
    loc_addrs = [locations[i] for i in range(len(locations))]
    cluster_point_addrs = [[loc_addrs[p_idx] for p_idx in c] for c in clusters]
    last_exit_point_idx = None

    for i, cluster_idx in enumerate(cluster_order):
        current_cluster_indices = clusters[cluster_idx]
        if not current_cluster_indices: continue
        if len(current_cluster_indices) == 1:
            cluster_path = current_cluster_indices
        else:
            if i == 0: # 第一個群集
                next_cluster_idx = cluster_order[i + 1]
                _, best_origin_idx, _ = find_closest_point_pair(gmaps_client, cluster_point_addrs[cluster_idx], cluster_point_addrs[next_cluster_idx], mode=mode)
                exit_point_idx = current_cluster_indices[best_origin_idx]
                
                path_candidate = solve_points_tsp_with_ga(current_cluster_indices, walking_matrix)
                if path_candidate[-1] != exit_point_idx:
                    path_candidate.reverse()
                cluster_path = path_candidate

            elif i == len(cluster_order) - 1: # 最後一個群集
                _, _, best_dest_idx = find_closest_point_pair(gmaps_client, [loc_addrs[last_exit_point_idx]], cluster_point_addrs[cluster_idx], mode=mode)
                entry_point_idx = current_cluster_indices[best_dest_idx]
                
                path_candidate = solve_points_tsp_with_ga(current_cluster_indices, walking_matrix)
                if path_candidate[0] != entry_point_idx:
                    path_candidate.reverse()
                cluster_path = path_candidate
            
            else: # 中間群集
                 _, _, best_dest_idx = find_closest_point_pair(gmaps_client, [loc_addrs[last_exit_point_idx]], cluster_point_addrs[cluster_idx], mode=mode)
                 entry_point_idx = current_cluster_indices[best_dest_idx]
                 next_cluster_idx = cluster_order[i + 1]
                 _, best_origin_idx, _ = find_closest_point_pair(gmaps_client, cluster_point_addrs[cluster_idx], cluster_point_addrs[next_cluster_idx], mode=mode)
                 exit_point_idx = current_cluster_indices[best_origin_idx]

                 # 簡化：先產生最佳開放路徑，再對齊起點和終點
                 path_candidate = solve_points_tsp_with_ga(current_cluster_indices, walking_matrix)
                 if path_candidate[0] != entry_point_idx:
                     path_candidate.reverse()
                 
                 # 確保終點在最後
                 if exit_point_idx in path_candidate:
                     path_candidate.remove(exit_point_idx)
                 path_candidate.append(exit_point_idx)
                 cluster_path = path_candidate

        # 合併路徑
        for point_idx in cluster_path:
            if point_idx not in final_route_indices:
                final_route_indices.append(point_idx)
        
        if final_route_indices:
            last_exit_point_idx = final_route_indices[-1]

    return final_route_indices