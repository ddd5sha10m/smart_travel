
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
                                                             )
                
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

'''
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
               variable_type='float',
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
'''
def solve_cluster_tsp_bruteforce(cluster_matrix):
    """
    使用暴力窮舉法，找到拜訪所有群集的絕對最佳順序。
    """
    num_clusters = len(cluster_matrix)
    if num_clusters <= 2:
        return list(range(num_clusters))

    min_path_duration = float('inf')
    best_path = []
    
    # 產生從 0 到 n-1 的所有排列組合
    for permutation in itertools.permutations(range(num_clusters)):
        current_duration = 0
        for i in range(num_clusters - 1):
            start_node = permutation[i]
            end_node = permutation[i+1]
            current_duration += cluster_matrix[start_node][end_node]
        
        if current_duration < min_path_duration:
            min_path_duration = current_duration
            best_path = list(permutation)
            
    print(f"窮舉法找到的最佳群集順序: {best_path}")
    return best_path


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
                                                 )
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

'''
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
'''
def solve_points_tsp_with_ga(point_indices, full_walking_matrix):
    """
    使用基因演算法，解決一組指定地點（開放路徑）的 TSP 問題。
    (V2 - 新增重複懲罰，確保路線合法)
    """
    num_points = len(point_indices)
    if num_points <= 2:
        return point_indices

    # --- 建立子矩陣 (邏輯不變) ---
    sub_matrix = np.full((num_points, num_points), 0)
    index_map = {i: original_idx for i, original_idx in enumerate(point_indices)}
    for i in range(num_points):
        for j in range(num_points):
            sub_matrix[i][j] = full_walking_matrix[index_map[i]][index_map[j]]
    
    # --- 改造適應度函式 ---
    def fitness_function(solution):
        
        # --- 新增的懲罰邏輯 ---
        # 檢查 solution 中是否有重複的數字。
        # set(solution) 會移除重複項，如果長度變短，代表有重複。
        if len(set(solution)) < num_points:
            # 如果有重複，給予一個極大的懲罰值，讓此解被淘汰
            return float('inf')
        # --- 懲罰邏輯結束 ---

        # (計算總時間的邏輯不變)
        total_time = 0
        for i in range(num_points - 1):
            start_node = int(solution[i])
            end_node = int(solution[i+1])
            total_time += sub_matrix[start_node][end_node]
        return total_time

    # (演算法參數和執行的部分不變)
    varbound = np.array([[0, num_points-1]] * num_points)
    algorithm_param = {
        'max_num_iteration': 1500,        # 增加迭代次數，給予更多演化時間
        'population_size': 100,           # 增加族群大小，提升多樣性
        'mutation_probability': 0.1,      # 保持或微調突變率
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'one_point',
        'max_iteration_without_improv': None
    }
    model = ga(function=fitness_function, dimension=num_points, variable_type='int', variable_boundaries=varbound, algorithm_parameters=algorithm_param, convergence_curve=False, progress_bar=False)
    model.run()
    
    best_local_route = model.best_variable.astype(int)
    best_global_route = [index_map[i] for i in best_local_route]
    
    return best_global_route

'''
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
                    path = solve_cluster_tsp_bruteforce(start_candidate, exit_point_idx, middle_points, walking_matrix)
                    
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
            cluster_path = solve_cluster_tsp_bruteforce(entry_point_idx, exit_point_idx, internal_points, walking_matrix)
            last_exit_point_idx = exit_point_idx

        # --- 合併路徑 ---
        for point_idx in cluster_path:
            if point_idx not in final_route_indices:
                final_route_indices.append(point_idx)

    return final_route_indices
'''
# (用這個新版本，完整替換 routing_logic.py 中的 create_final_itinerary)
def create_final_itinerary(locations, clusters, cluster_order, walking_matrix, gmaps_client, mode='driving'):
    """
    建立最終的點對點行程 (V4 - Fixed Path Correction)。
    """
    final_route_indices = []
    loc_addrs = [locations[i] for i in range(len(locations))]
    cluster_point_addrs = [[loc_addrs[p_idx] for p_idx in c] for c in clusters]
    last_exit_point_idx = None

    for i, cluster_idx in enumerate(cluster_order):
        current_cluster_indices = clusters[cluster_idx]
        
        if len(current_cluster_indices) <= 1:
             cluster_path = current_cluster_indices
             if cluster_path: # 如果群集不是空的
                last_exit_point_idx = cluster_path[0]
        elif i == 0:
            # 優化第一個群集：找出一個最佳路徑，使其離開點最靠近下一個群集
            next_cluster_idx = cluster_order[i + 1]
            _, best_origin_idx, _ = find_closest_point_pair(gmaps_client, cluster_point_addrs[cluster_idx], cluster_point_addrs[next_cluster_idx], mode=mode)
            exit_point_idx = current_cluster_indices[best_origin_idx]
            internal_points = [p for p in current_cluster_indices if p != exit_point_idx]
            
            # 使用開放路徑 GA 來規劃從起點到離開點的路徑
            path_candidate = solve_points_tsp_with_ga(internal_points + [exit_point_idx], walking_matrix)
            if path_candidate[-1] != exit_point_idx: path_candidate.reverse()
            cluster_path = path_candidate
            last_exit_point_idx = exit_point_idx
            
        elif i == len(cluster_order) - 1:
            # 優化最後一個群集：從最靠近上一個群集的進入點開始，走一條最短的開放路徑
            prev_cluster_addrs = [loc_addrs[last_exit_point_idx]]
            _, _, best_dest_idx = find_closest_point_pair(gmaps_client, prev_cluster_addrs, cluster_point_addrs[cluster_idx], mode=mode)
            entry_point_idx = current_cluster_indices[best_dest_idx]
            internal_points = [p for p in current_cluster_indices if p != entry_point_idx]
            
            # 使用開放路徑 GA
            path_candidate = solve_points_tsp_with_ga([entry_point_idx] + internal_points, walking_matrix)
            if path_candidate[0] != entry_point_idx: path_candidate.reverse()
            cluster_path = path_candidate
            
        else:
            # --- 最關鍵的修正：處理中間群集 ---
            # 1. 嚴格計算進入點和離開點
            prev_cluster_addrs = [loc_addrs[last_exit_point_idx]]
            _, _, best_dest_idx = find_closest_point_pair(gmaps_client, prev_cluster_addrs, cluster_point_addrs[cluster_idx], mode=mode)
            entry_point_idx = current_cluster_indices[best_dest_idx]

            next_cluster_idx = cluster_order[i + 1]
            _, best_origin_idx, _ = find_closest_point_pair(gmaps_client, cluster_point_addrs[cluster_idx], cluster_point_addrs[next_cluster_idx], mode=mode)
            exit_point_idx = current_cluster_indices[best_origin_idx]

            # 2. 找出需要被安排順序的中間點
            middle_points = [p for p in current_cluster_indices if p != entry_point_idx and p != exit_point_idx]
            
            # 3. 使用我們全新的「固定路徑」解算器
            print(f"正在規劃固定路徑: 從 {entry_point_idx} 到 {exit_point_idx}...")
            cluster_path = solve_fixed_path_tsp_with_ga(entry_point_idx, exit_point_idx, middle_points, walking_matrix)
            
            # 4. 嚴格遵守計算出的離開點
            last_exit_point_idx = exit_point_idx

        # 合併路徑
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

def solve_fixed_path_tsp_with_ga(start_point_idx, end_point_idx, middle_points_indices, full_walking_matrix):
    """
    使用基因演算法，解決帶有固定起點和終點的 TSP 問題。
    """
    if not middle_points_indices:
        return [start_point_idx, end_point_idx] if start_point_idx != end_point_idx else [start_point_idx]

    # 參與排序的只有中間點
    points_to_permute = middle_points_indices
    num_points = len(points_to_permute)

    # 建立子矩陣和索引映射，這與 solve_points_tsp_with_ga 類似
    sub_matrix = np.full((num_points, num_points), 0)
    index_map = {i: original_idx for i, original_idx in enumerate(points_to_permute)}
    for i in range(num_points):
        for j in range(num_points):
            sub_matrix[i][j] = full_walking_matrix[index_map[i]][index_map[j]]

    def fitness_function(solution):
        # 檢查重複性
        if len(set(solution)) < num_points:
            return float('inf')
        
        # 將 solution (可能是 floats) 轉換為整數列表
        path = [int(p) for p in solution]
        
        # 計算總時間
        total_time = 0
        # 1. 從固定起點到第一個中間點
        total_time += full_walking_matrix[start_point_idx][index_map[path[0]]]
        
        # 2. 中間點之間的路徑 (FIXED)
        for i in range(num_points - 1):
            start_node = path[i]
            end_node = path[i+1]
            total_time += sub_matrix[start_node][end_node]
            
        # 3. 從最後一個中間點到固定終點
        total_time += full_walking_matrix[index_map[path[-1]]][end_point_idx]

        return total_time

    varbound = np.array([[0, num_points - 1]] * num_points)
    algorithm_param = {
        'max_num_iteration': 1500,        # 增加迭代次數，給予更多演化時間
        'population_size': 100,           # 增加族群大小，提升多樣性
        'mutation_probability': 0.1,      # 保持或微調突變率
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'one_point',
        'max_iteration_without_improv': None
    }
    model = ga(function=fitness_function, dimension=num_points, variable_type='int', variable_boundaries=varbound, algorithm_parameters=algorithm_param, convergence_curve=False, progress_bar=False)
    model.run()
    
    best_local_middle_route = model.best_variable.astype(int)
    best_global_middle_route = [index_map[i] for i in best_local_middle_route]
    
    # 組合最終路徑：起點 + 優化後的中間點 + 終點
    return [start_point_idx] + best_global_middle_route + [end_point_idx]
