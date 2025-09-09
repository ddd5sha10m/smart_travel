import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import itertools

def calculate_cluster_matrix(gmaps_client, locations_list, clusters, mode='driving'):
    """
    計算群集之間的交通時間矩陣。

    Args:
        gmaps_client: googlemaps.Client 的實例。
        locations_list (list): 原始的地點名稱列表。
        clusters (list of lists): 分群結果，例如 [[0, 3], [1], [2]]。
        mode (str): 交通方式，例如 'driving', 'transit'。

    Returns:
        np.array: 一個 N x N 的 NumPy 陣列，N 是群集的數量。
                  matrix[i][j] 代表從 cluster i 到 cluster j 的最短交通時間（秒）。
    """
    num_clusters = len(clusters)
    cluster_matrix = np.full((num_clusters, num_clusters), -1, dtype=int)

    # 為了減少 API 呼叫，我們先將索引對應到實際的地址
    cluster_addresses = []
    for cluster in clusters:
        addresses = [locations_list[i] for i in cluster]
        cluster_addresses.append(addresses)

    # 遍歷每一對群集來計算它們之間的交通時間
    for i in range(num_clusters):
        for j in range(num_clusters):
            if i == j:
                cluster_matrix[i][j] = 0
                continue

            origins = cluster_addresses[i]
            destinations = cluster_addresses[j]

            try:
                # 取得 i 和 j 兩個群集之間所有點對點的交通時間
                matrix_result = gmaps_client.distance_matrix(origins=origins,
                                                             destinations=destinations,
                                                             mode=mode,
                                                             language="zh-TW")
                
                # 從所有可能的路徑中，找出時間最短的那一條
                min_duration = float('inf')
                for row in matrix_result['rows']:
                    for element in row['elements']:
                        if element['status'] == 'OK':
                            min_duration = min(min_duration, element['duration']['value'])
                
                if min_duration != float('inf'):
                    cluster_matrix[i][j] = min_duration
                else:
                    print(f"警告: 在群集 {i} 和 {j} 之間找不到有效的 {mode} 路徑。")

            except Exception as e:
                print(f"計算群集 {i} 到 {j} 的矩陣時出錯: {e}")

    return cluster_matrix


def solve_cluster_tsp_with_ga(cluster_matrix):
    """
    使用基因演算法解決群集順序的 TSP 問題。

    Args:
        cluster_matrix (np.array): 群集間的交通時間矩陣。

    Returns:
        list: 最佳的群集拜訪順序，例如 [1, 0, 2]。
    """
    num_clusters = len(cluster_matrix)

    # 定義基因演算法的「適應度函式 (Fitness Function)」
    # 我們的目標是找到總路徑時間最短的解，所以適應度就是總時間
    def fitness_function(solution):
        total_time = 0
        for i in range(num_clusters - 1):
            start_cluster = int(solution[i])
            end_cluster = int(solution[i+1])
            total_time += cluster_matrix[start_cluster][end_cluster]
        return total_time

    # 設定演算法參數
    varbound = np.array([[0, num_clusters-1]] * num_clusters)
    algorithm_param = {
        'max_num_iteration': 500,
        'population_size': 50,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'one_point', # Corrected to a valid option
        'max_iteration_without_improv': None
    }

    # 建立並執行模型
    model = ga(function=fitness_function,
               dimension=num_clusters,
               variable_type='int',
               variable_boundaries=varbound,
               variable_type_mixed=None,
               function_timeout=10,
               algorithm_parameters=algorithm_param,
               convergence_curve=False,
               progress_bar=False) # 關閉進度條，避免洗版
    
    model.run()

    # 獲取最佳解 (最佳路徑)
    best_route = model.best_variable.astype(int).tolist()
    
    print(f"基因演算法找到的最佳群集順序: {best_route}")
    return best_route

def find_closest_point_pair(gmaps_client, origins_addrs, dests_addrs, mode='driving'):
    """
    找到兩個地點集合之間交通時間最短的一對點。

    Returns:
        tuple: (最短時間(秒), 最佳起點索引, 最佳終點索引)
               索引是相對於 origins_addrs 和 dests_addrs 列表的。
    """
    matrix_result = gmaps_client.distance_matrix(origins=origins_addrs,
                                                 destinations=dests_addrs,
                                                 mode=mode,
                                                 language="zh-TW")
    min_duration = float('inf')
    best_origin_idx, best_dest_idx = -1, -1

    for i, row in enumerate(matrix_result['rows']):
        for j, element in enumerate(row['elements']):
            if element['status'] == 'OK':
                duration = element['duration']['value']
                if duration < min_duration:
                    min_duration = duration
                    best_origin_idx = i
                    best_dest_idx = j
    
    return min_duration, best_origin_idx, best_dest_idx


def solve_intra_cluster_tsp(entry_point_idx, exit_point_idx, internal_points_indices, walking_matrix):
    """
    使用暴力窮舉法，解決群集內部的 TSP 問題。
    找出從 entry_point 到 exit_point，走完所有 internal_points 的最短路徑。
    """
    if not internal_points_indices:
        return [entry_point_idx, exit_point_idx] if entry_point_idx != exit_point_idx else [entry_point_idx]

    best_path = []
    min_path_duration = float('inf')

    # 產生所有內部點的排列組合
    for permutation in itertools.permutations(internal_points_indices):
        current_path = [entry_point_idx] + list(permutation) + [exit_point_idx]
        current_duration = 0
        
        # 計算此排列的路徑總長
        for i in range(len(current_path) - 1):
            start_node = current_path[i]
            end_node = current_path[i+1]
            current_duration += walking_matrix[start_node][end_node]
        
        if current_duration < min_path_duration:
            min_path_duration = current_duration
            best_path = current_path
            
    return best_path


def create_final_itinerary(locations, clusters, cluster_order, walking_matrix, gmaps_client, mode='driving'):
    """
    建立最終的點對點行程 (V2 - Refactored for Correctness)。
    """
    final_route_indices = []
    loc_addrs = [locations[i] for i in range(len(locations))]
    cluster_point_addrs = [[loc_addrs[p_idx] for p_idx in c] for c in clusters]
    last_exit_point_idx = None

    for i, cluster_idx in enumerate(cluster_order):
        current_cluster_indices = clusters[cluster_idx]
        
        # --- 全新的、分情況的最佳化邏輯 ---

        if i == 0:  # --- Case 1: The VERY FIRST cluster ---
            print("優化第一個群集...")
            # 目標：找出一個最佳路徑，使其離開點 (exit_point) 最靠近下一個群集。
            next_cluster_idx = cluster_order[i + 1]
            _, best_origin_idx, _ = find_closest_point_pair(
                gmaps_client, cluster_point_addrs[cluster_idx], cluster_point_addrs[next_cluster_idx], mode=mode)
            exit_point_idx = current_cluster_indices[best_origin_idx]

            # 找出除了離開點以外的其他點
            internal_points = [p for p in current_cluster_indices if p != exit_point_idx]
            
            if not internal_points: # 如果群集只有一個點
                cluster_path = [exit_point_idx]
            else:
                # 暴力窮舉：嘗試從每一個 internal_point 出發，到 exit_point 結束的最佳路徑
                best_path_for_first_cluster = []
                min_duration = float('inf')

                for start_candidate in internal_points:
                    middle_points = [p for p in internal_points if p != start_candidate]
                    path = solve_intra_cluster_tsp(start_candidate, exit_point_idx, middle_points, walking_matrix)
                    
                    # 計算這條路徑的總長
                    path_duration = sum(walking_matrix[path[k]][path[k+1]] for k in range(len(path) - 1))
                    
                    if path_duration < min_duration:
                        min_duration = path_duration
                        best_path_for_first_cluster = path
                
                cluster_path = best_path_for_first_cluster
            
            last_exit_point_idx = exit_point_idx

        elif i == len(cluster_order) - 1:  # --- Case 3: The VERY LAST cluster ---
            print("優化最後一個群集...")
            # 目標：從最靠近上一個群集的進入點 (entry_point) 開始，走一條最短的開放路徑。
            prev_cluster_addrs = [loc_addrs[last_exit_point_idx]]
            _, _, best_dest_idx = find_closest_point_pair(
                gmaps_client, prev_cluster_addrs, cluster_point_addrs[cluster_idx], mode=mode)
            entry_point_idx = current_cluster_indices[best_dest_idx]

            internal_points = [p for p in current_cluster_indices if p != entry_point_idx]

            # 從進入點開始，規劃一條走完剩下點的最短開放路徑
            open_path_internal = solve_open_tsp_bruteforce(internal_points, walking_matrix)
            cluster_path = [entry_point_idx] + open_path_internal
            
        else:  # --- Case 2: Intermediate clusters ---
            print("優化中間群集...")
            # 目標：規劃一條從 entry_point 到 exit_point 的最短路徑。
            # 1. 決定進入點 (修正 Bug：使用 mode 而不是 'walking')
            prev_cluster_addrs = [loc_addrs[last_exit_point_idx]]
            _, _, best_dest_idx = find_closest_point_pair(
                gmaps_client, prev_cluster_addrs, cluster_point_addrs[cluster_idx], mode=mode)
            entry_point_idx = current_cluster_indices[best_dest_idx]

            # 2. 決定離開點
            next_cluster_idx = cluster_order[i + 1]
            _, best_origin_idx, _ = find_closest_point_pair(
                gmaps_client, cluster_point_addrs[cluster_idx], cluster_point_addrs[next_cluster_idx], mode=mode)
            exit_point_idx = current_cluster_indices[best_origin_idx]

            internal_points = [p for p in current_cluster_indices if p != entry_point_idx and p != exit_point_idx]
            cluster_path = solve_intra_cluster_tsp(entry_point_idx, exit_point_idx, internal_points, walking_matrix)
            last_exit_point_idx = exit_point_idx

        # --- 合併路徑 ---
        for point_idx in cluster_path:
            if point_idx not in final_route_indices:
                final_route_indices.append(point_idx)

    return final_route_indices

def solve_open_tsp_bruteforce(point_indices, matrix):
    """
    使用暴力窮舉法解決一個開放路徑的 TSP 問題（不需回到起點）。
    適用於單一群集內部路徑規劃。

    Args:
        point_indices (list): 需要被排序的地點索引列表。
        matrix (list of lists): 包含這些地點之間交通時間的完整矩陣。

    Returns:
        list: 最佳化的地點索引順序。
    """
    if len(point_indices) <= 2:
        return point_indices

    best_path = []
    min_total_duration = float('inf')

    # 窮舉所有可能的排列
    for permutation in itertools.permutations(point_indices):
        current_duration = 0
        # 計算當前排列路徑的總時間
        for i in range(len(permutation) - 1):
            start_node = permutation[i]
            end_node = permutation[i+1]
            current_duration += matrix[start_node][end_node]
        
        # 如果找到更短的路徑，就更新紀錄
        if current_duration < min_total_duration:
            min_total_duration = current_duration
            best_path = list(permutation)
            
    return best_path